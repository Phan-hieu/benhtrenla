function predict() {
    var formData = new FormData();
    var file = document.getElementById('file').files[0];
    formData.append('file', file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    }).then(function (response) {
        return response.json();
    }).then(function (data) {
        var resultH2 = document.getElementById('result');
        resultH2.textContent = 'Loại bệnh được dự đoán: ' + data.class;
    }).catch(function (error) {
        console.error(error);
    });
}

document.querySelector('form').addEventListener('submit', function (event) {
    event.preventDefault();
    predict();
}); 
