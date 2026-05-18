document.addEventListener("DOMContentLoaded", () => {
    
    // Core DOM Elements
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const previewArea = document.getElementById("preview-area");
    const previewImg = document.getElementById("preview-img");
    const fileNameText = document.getElementById("file-name");
    const fileSizeText = document.getElementById("file-size");
    const removeBtn = document.getElementById("remove-btn");
    const analyzeBtn = document.getElementById("analyze-btn");
    const sampleCards = document.querySelectorAll(".sample-card");
    
    // Mode Switcher Elements
    const modeSimBtn = document.getElementById("mode-sim");
    const modeRealBtn = document.getElementById("mode-real");
    
    // Results DOM Elements
    const resultsPlaceholder = document.getElementById("results-placeholder");
    const resultsLoader = document.getElementById("results-loader");
    const loaderProgressText = document.getElementById("loader-progress");
    const diagnosticReport = document.getElementById("diagnostic-report");
    
    const reportBadge = document.getElementById("report-badge");
    const reportTitle = document.getElementById("report-title");
    const reportConfidenceFill = document.getElementById("report-confidence-fill");
    const reportConfidenceText = document.getElementById("report-confidence-text");
    const reportType = document.getElementById("report-type");
    const reportSeverity = document.getElementById("report-severity");
    const reportSymptomsList = document.getElementById("report-symptoms");
    const reportDesc = document.getElementById("report-desc");
    const reportTreatment = document.getElementById("report-treatment");

    let selectedFile = null;
    let diagnosticMode = "sim"; // Default to high-fidelity Showcase Simulation

    // --- Diagnostic Mode Switcher Logic ---
    modeSimBtn.addEventListener("click", () => {
        diagnosticMode = "sim";
        modeSimBtn.classList.add("active");
        modeSimBtn.style.background = "var(--primary-gradient)";
        modeSimBtn.style.color = "var(--text-light)";
        modeSimBtn.style.boxShadow = "0 0 10px rgba(16, 185, 129, 0.25)";
        
        modeRealBtn.classList.remove("active");
        modeRealBtn.style.background = "transparent";
        modeRealBtn.style.color = "var(--text-muted)";
        modeRealBtn.style.boxShadow = "none";
    });

    modeRealBtn.addEventListener("click", () => {
        diagnosticMode = "real";
        modeRealBtn.classList.add("active");
        modeRealBtn.style.background = "var(--primary-gradient)";
        modeRealBtn.style.color = "var(--text-light)";
        modeRealBtn.style.boxShadow = "0 0 10px rgba(16, 185, 129, 0.25)";
        
        modeSimBtn.classList.remove("active");
        modeSimBtn.style.background = "transparent";
        modeSimBtn.style.color = "var(--text-muted)";
        modeSimBtn.style.boxShadow = "none";
    });

    // --- Drag & Drop Operations ---
    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add("dragover");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove("dragover");
        }, false);
    });

    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileSelection(files[0]);
        }
    });

    dropZone.addEventListener("click", () => {
        fileInput.click();
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    function handleFileSelection(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please upload a valid image file (PNG, JPG, or JPEG).");
            return;
        }

        selectedFile = file;
        fileNameText.innerText = file.name;
        fileSizeText.innerText = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            dropZone.style.display = "none";
            previewArea.style.display = "block";
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);

        sampleCards.forEach(c => c.style.borderColor = "var(--border-color)");
    }

    removeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        resetUploadZone();
    });

    function resetUploadZone() {
        selectedFile = null;
        fileInput.value = "";
        previewImg.src = "";
        dropZone.style.display = "block";
        previewArea.style.display = "none";
        analyzeBtn.disabled = true;
        
        diagnosticReport.style.display = "none";
        resultsLoader.style.display = "none";
        resultsPlaceholder.style.display = "block";
    }

    // --- Action & Analysis Routines ---
    analyzeBtn.addEventListener("click", () => {
        if (!selectedFile) return;

        const formData = new FormData();
        formData.append("image", selectedFile);

        triggerLoadingSequence();

        fetch(`/predict?mode=${diagnosticMode}`, {
            method: "POST",
            body: formData
        })
        .then(response => {
            if (!response.ok) throw new Error("Diagnostic server returned an error.");
            return response.json();
        })
        .then(data => {
            setTimeout(() => {
                renderDiagnosticReport(data);
            }, 1000);
        })
        .catch(err => {
            alert(`Analysis failed: ${err.message}`);
            resultsLoader.style.display = "none";
            resultsPlaceholder.style.display = "block";
        });
    });

    sampleCards.forEach(card => {
        card.addEventListener("click", () => {
            const className = card.getAttribute("data-class");
            
            sampleCards.forEach(c => c.style.borderColor = "var(--border-color)");
            card.style.borderColor = "var(--primary)";

            resetUploadZone();
            
            const clickedImgSrc = card.querySelector("img").src;
            previewImg.src = clickedImgSrc;
            fileNameText.innerText = `Sample: ${className.split("___").join(" - ")}`;
            fileSizeText.innerText = "Local dataset file";
            dropZone.style.display = "none";
            previewArea.style.display = "block";

            triggerLoadingSequence();
            loaderProgressText.innerText = "Scanning plant leaf library...";

            fetch(`/predict-sample/${className}?mode=${diagnosticMode}`)
            .then(response => {
                if (!response.ok) throw new Error("Failed to process local sample.");
                return response.json();
            })
            .then(data => {
                setTimeout(() => {
                    renderDiagnosticReport(data);
                }, 1000);
            })
            .catch(err => {
                alert(`Sample diagnosis failed: ${err.message}`);
                resultsLoader.style.display = "none";
                resultsPlaceholder.style.display = "block";
            });
        });
    });

    function triggerLoadingSequence() {
        resultsPlaceholder.style.display = "none";
        diagnosticReport.style.display = "none";
        resultsLoader.style.display = "block";
        
        const progressSteps = [
            "Normalizing image color channels...",
            "Constructing convolution feature map...",
            "Running MobileNetV2 spatial filter weightings...",
            "Matching classification vectors...",
            "Retrieving plant health diagnosis..."
        ];

        let currentStepIdx = 0;
        loaderProgressText.innerText = progressSteps[0];

        const interval = setInterval(() => {
            currentStepIdx++;
            if (currentStepIdx < progressSteps.length) {
                loaderProgressText.innerText = progressSteps[currentStepIdx];
            } else {
                clearInterval(interval);
            }
        }, 300);
    }

    function renderDiagnosticReport(data) {
        resultsLoader.style.display = "none";
        diagnosticReport.style.display = "block";

        reportTitle.innerText = data.display_name;
        reportType.innerText = data.type;
        reportSeverity.innerText = data.severity;
        reportDesc.innerText = data.description;
        reportTreatment.innerText = data.treatment;

        reportSymptomsList.innerHTML = "";
        data.symptoms.forEach(symptom => {
            const li = document.createElement("li");
            li.className = "report-list-item";
            li.innerText = symptom;
            reportSymptomsList.appendChild(li);
        });

        const color = data.color;
        reportBadge.innerText = data.status;
        reportBadge.style.color = color;
        reportBadge.style.borderColor = color.replace(")", ", 0.25)").replace("var(", "rgba(");
        reportBadge.style.background = color.replace(")", ", 0.1)").replace("var(", "rgba(");
        
        reportSeverity.style.color = color;

        reportTreatment.className = "treatment-box";
        if (data.severity === "Critical") {
            reportTreatment.classList.add("danger-box");
        } else if (data.severity === "High" || data.severity === "Medium") {
            reportTreatment.classList.add("warning-box");
        }

        reportConfidenceText.innerText = data.confidence;
        reportConfidenceFill.style.width = "0%";
        reportConfidenceFill.style.background = `linear-gradient(135deg, ${color}, rgba(255,255,255,0.15))`;
        
        setTimeout(() => {
            reportConfidenceFill.style.width = data.confidence;
        }, 50);
    }
});
