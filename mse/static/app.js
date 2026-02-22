// Task 3.3.3

const BASE_URL = ''; 
// const BASE_URL = 'http://localhost:8000'; // для интеграции с бэкендом

let isTraining = false;
let mockInterval = null;
let currentTrial = 0;
let totalTrials = 0;


const trainingForm = document.getElementById('trainingForm');
const startButton = document.getElementById('startButton');
const cancelButton = document.getElementById('cancelButton');
const monitoringBlock = document.getElementById('monitoringBlock');
const resultsBlock = document.getElementById('resultsBlock');
const randomIterationsGroup = document.getElementById('randomIterationsGroup');
const algorithmSelect = document.getElementById('algorithm');
const nIterInput = document.getElementById('nIter');
const statusMessage = document.getElementById('statusMessage');
const currentModelSpan = document.getElementById('currentModel');
const totalModelsSpan = document.getElementById('totalModels');
const progressFill = document.getElementById('progressFill');
const logsArea = document.getElementById('logsArea');
const resultChart = document.getElementById('resultChart');
const downloadModel = document.getElementById('downloadModel');



algorithmSelect.addEventListener('change', function() {
    if (this.value === 'random') {
        randomIterationsGroup.style.display = 'block';
        nIterInput.required = true;
    } else {
        randomIterationsGroup.style.display = 'none';
        nIterInput.required = false;
    }
});

if (algorithmSelect.value === 'random') {
    randomIterationsGroup.style.display = 'block';
} else {
    randomIterationsGroup.style.display = 'none';
}
function collectFormData() {
    const formData = {
        dataset_path: document.getElementById('datasetPath').value,
        search_algorithm: algorithmSelect.value,
        base_model: document.getElementById('yoloVersion').value,
        hyperparameters: {
            epochs: [parseInt(document.getElementById('epochs').value)],
            lr0: [parseFloat(document.getElementById('learningRate').value)],
            batch: [parseInt(document.getElementById('batchSize').value)]
        },
        early_stop_patience: 3,
        early_stop_delta: 0.15
    };

    //n_iter для random search
    if (algorithmSelect.value === 'random') {
        formData.n_iter = parseInt(nIterInput.value);
    }

    return formData;
}

function mockStartTraining(data) {
    return new Promise((resolve) => {
        console.log('Отправка данных:', data);

        startButton.disabled = true;
        cancelButton.disabled = false;
        isTraining = true;

        monitoringBlock.style.display = 'block';
        resultsBlock.style.display = 'none';

        if (data.search_algorithm === 'grid') {
            totalTrials = 4; //пример
        } else {
            totalTrials = data.n_iter || 5;
        }
        totalModelsSpan.textContent = totalTrials;

        setTimeout(() => {
            console.log('Ответ:', { status: 'started' });
            resolve({ status: 'started' });

            startMockProgress();
        }, 1000);
    });
}

function createMockFiles() {
    console.log('Используем файлы из static/mocks/');
}

function startMockProgress() {
    currentTrial = 0;
    updateStatusMessage();

    if (mockInterval) {
        clearInterval(mockInterval);
    }

    mockInterval = setInterval(() => {
        currentTrial++;
        if (currentTrial <= totalTrials) {
            updateStatusMessage();
            fetchMockStatus();
            fetchMockLogs();
        }

        if (currentTrial >= totalTrials) {
            setTimeout(() => {
                completeTraining();
            }, 2000);
        }
    }, 3000);
}

//обновление статуса
function updateStatusMessage() {
    currentModelSpan.textContent = currentTrial;
    const percent = (currentTrial / totalTrials) * 100;
    progressFill.style.width = percent + '%';

    if (currentTrial < totalTrials) {
        statusMessage.innerHTML = `Обучается модель <span id="currentModel">${currentTrial}</span> из <span id="totalModels">${totalTrials}</span>`;
    }
}

// Task 3.2.2:
async function fetchMockStatus() {
    try {
        const response = await fetch(`${BASE_URL}/static/mocks/dummy_status.json`);
        const statusData = await response.json();

        console.log('Статус:', statusData);

        if (statusData.status === 'completed') {
            completeTraining();
        }
    } catch (error) {
        console.warn('Не удалось загрузить dummy_status.json, используем эмуляцию');
        if (currentTrial >= totalTrials) {
            completeTraining();
        }
    }
}

// Task 3.2.3
async function fetchMockLogs() {
    try {
        const response = await fetch(`${BASE_URL}/static/mocks/dummy_logs.txt?t=${Date.now()}`);
        const logsText = await response.text();

        const timestamp = new Date().toLocaleTimeString();
        const newLogs = `[${timestamp}] Эпоха ${currentTrial}/${totalTrials}\n${logsText}\n\n`;

        logsArea.value += newLogs;
        logsArea.scrollTop = logsArea.scrollHeight;
    } catch (error) {
        console.warn('Не удалось загрузить dummy_logs.txt, используем эмуляцию');
        emulateLogs();
    }
}

function emulateLogs() {
    const timestamp = new Date().toLocaleTimeString();
    const logTemplates = [
        `Epoch ${currentTrial}: train_loss=1.234, val_loss=2.345, mAP50=0.567`,
        `Epoch ${currentTrial}: learning_rate=0.001, batch_size=16`,
        `Epoch ${currentTrial}: saving checkpoint to runs/automl/trial_${currentTrial}`,
        `Epoch ${currentTrial}: best model updated (mAP50=0.678)`
    ];

    const randomLog = logTemplates[Math.floor(Math.random() * logTemplates.length)];
    const newLogs = `[${timestamp}] ${randomLog}\n`;

    logsArea.value += newLogs;
    logsArea.scrollTop = logsArea.scrollHeight;
}

// Task 3.3

function completeTraining() {
    if (!isTraining) return;

    console.log('Обучение завершено!');

    if (mockInterval) {
        clearInterval(mockInterval);
        mockInterval = null;
    }

    isTraining = false;
    startButton.disabled = false;
    cancelButton.disabled = true;

    statusMessage.innerHTML = `Обучение завершено! Всего моделей: ${totalTrials}`;
    progressFill.style.width = '100%';

    showResults();
}

function showResults() {
    const timestamp = new Date().toLocaleTimeString();
    logsArea.value += `\n[${timestamp}] AutoML эксперимент завершен!\n`;
    logsArea.value += `[${timestamp}] Результаты сохранены в runs/detect/automl/\n`;
    logsArea.scrollTop = logsArea.scrollHeight;

    resultsBlock.style.display = 'block';

    resultChart.src = `${BASE_URL}/static/assets/test_chart.png`;
    downloadModel.href = `${BASE_URL}/static/assets/dummy.pt`;
}

function cancelTraining() {
    if (mockInterval) {
        clearInterval(mockInterval);
        mockInterval = null;
    }

    isTraining = false;
    startButton.disabled = false;
    cancelButton.disabled = true;

    logsArea.value += `\n[${new Date().toLocaleTimeString()}] Обучение остановлено пользователем\n`;

    console.log('Обучение отменено');
}


trainingForm.addEventListener('submit', async (e) => {
    e.preventDefault(); 
    const formData = collectFormData();
    console.log('Собранный JSON:', JSON.stringify(formData, null, 2));

    try {
        const response = await mockStartTraining(formData);
        console.log('Ответ от сервера:', response);

    } catch (error) {
        console.error('Ошибка:', error);
        alert('Произошла ошибка при запуске обучения');
        startButton.disabled = false;
        cancelButton.disabled = true;
    }
});

cancelButton.addEventListener('click', () => {
    cancelTraining();
});
