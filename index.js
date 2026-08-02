import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.8.0'

// By default the library looks for a local copy under /models/ before trying the
// Hub. Vite answers any unknown path with index.html and a 200, so that check
// "succeeds" and JSON.parse then chokes on '<!DOCTYPE html>'. Skip it and go
// straight to the Hub.
env.allowLocalModels = false

// Reference the HTML elements that we will need
const status = document.getElementById('status')
const image = document.getElementById('image')
const detectObjectsButton = document.getElementById('detect-objects')
const imageContainer = document.getElementById('image-container')

// Build the pipeline lazily, on first click, instead of with a top-level await.
// Loading it up front blocks this whole module, so the button stays disabled and
// the page looks frozen for the ~20MB the first run has to fetch.
let detectorPromise = null

function getDetector() {
    detectorPromise ??= pipeline('object-detection', 'Xenova/yolos-tiny', {
        // Fires per file as the weights come down. Note this covers the Hub files
        // only — the ~10MB onnxruntime WASM is fetched by ORT itself and never
        // reaches this callback, which is why the first stretch stays quiet.
        progress_callback: (item) => {
            if (item.status === 'progress') {
                status.textContent = `Loading ${item.file}... ${Math.round(item.progress)}%`
            }
        }
    }).catch(error => {
        // Don't let one failed load poison every later click. Without this the
        // rejected promise stays cached, so every retry re-throws the original
        // error instantly without hitting the network again.
        detectorPromise = null
        throw error
    })

    return detectorPromise
}

// Enable Object Detection
detectObjectsButton.addEventListener('click', detectAndDrawObjects)
detectObjectsButton.disabled = false
status.textContent = 'Ready'

async function detectAndDrawObjects() {
    detectObjectsButton.disabled = true

    try {
        // Only the first click pays for this; afterwards the promise is reused.
        status.textContent = 'Loading runtime...'
        const detector = await getDetector()

        // Detect Objects
        status.textContent = 'Detecting...'
        const detectedObjects = await detector(image.src, {
            threshold: 0.95,
            percentage: true
        })

        // Clear boxes from any previous run, otherwise they stack up
        imageContainer.querySelectorAll('.bounding-box').forEach(box => box.remove())

        // Draw Detected Objects
        status.textContent = 'Drawing...'
        detectedObjects.forEach(obj => {
            drawObjectBox(obj)
        })

        status.textContent = 'Done!'
    } catch (error) {
        console.error(error)
        status.textContent = `Something went wrong: ${error.message}`
    } finally {
        detectObjectsButton.disabled = false
    }
}

// Helper function that draws boxes for every detected object in the image
// ⚠️ ️This function requires box coordinates to be in percentages  ️
function drawObjectBox(detectedObject) {
    const { label, score, box } = detectedObject
    const { xmax, xmin, ymax, ymin } = box

    // Generate a random color for the box
    const color = '#' + Math.floor(Math.random() * 0xFFFFFF).toString(16).padStart(6, 0)
    
    // Draw the box
    const boxElement = document.createElement('div')
    boxElement.className = 'bounding-box'
    Object.assign(boxElement.style, {
        borderColor: color,
        left: 100 * xmin + '%',
        top: 100 * ymin + '%',
        width: 100 * (xmax - xmin) + '%',
        height: 100 * (ymax - ymin) + '%',
    })

    // Draw label
    const labelElement = document.createElement('span')
    labelElement.textContent = `${label}: ${Math.floor(score * 100)}%`
    labelElement.className = 'bounding-box-label'
    labelElement.style.backgroundColor = color

    boxElement.appendChild(labelElement)
    imageContainer.appendChild(boxElement)
}