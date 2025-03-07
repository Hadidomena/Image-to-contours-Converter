function previewFile() {
    const file = document.querySelector('input[type=file]').files[0];
    const preview = document.getElementById('uploaded-image');
    const reader = new FileReader();

    reader.addEventListener('load', function () {
        preview.src = reader.result;
        updateRecommendedValues(file);
    }, false);

    if (file) {
        reader.readAsDataURL(file);
    }
}

function updateRecommendedValues(file) {
    const formData = new FormData();
    formData.append('file', file);

    fetch('/get_recommended_values', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('simplification_factor').value = data.recommended_simplification_factor;
        document.getElementById('low_threshold').value = data.recommended_low_threshold;
        document.getElementById('high_threshold').value = data.recommended_high_threshold;
    })
    .catch(error => {
        console.error('Error fetching recommended values:', error);
    });
}

function copyCurves() {
    const curvesText = document.getElementById('curves-text').innerText;
    navigator.clipboard.writeText(curvesText).then(() => {
        alert('Curves copied to clipboard!');
    }).catch(err => {
        alert('Failed to copy curves: ' + err);
    });
}

function proceedWithRecommended() {
    const form = document.getElementById('upload-form');
    const input = document.createElement('input');
    input.type = 'hidden';
    input.name = 'recommended';
    input.value = 'true';
    form.appendChild(input);
    form.submit();
}

function checkProgress() {
    fetch('/progress')
        .then(response => response.json())
        .then(data => {
            const progress = document.getElementById('progress');
            const progressText = document.getElementById('progress-text');
            progress.style.width = `${data.progress}%`;
            progressText.textContent = `Processing: ${Math.round(data.progress)}%`;
            
            if (data.progress < 100) {
                setTimeout(checkProgress, 500);
            }
        });
}

document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('upload-form').onsubmit = function() {
        const fileInput = document.querySelector('input[type=file]');
        if (fileInput.files[0].name.match(/\.(mp4|gif)$/)) {
            document.getElementById('progress-container').style.display = 'block';
            setTimeout(checkProgress, 500);
        }
    };
});