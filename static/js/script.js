function predict() {
    var size = document.getElementById('size').value;
    var weight = document.getElementById('weight').value;
    var sweetness = document.getElementById('sweetness').value;
    var crunchiness = document.getElementById('crunchiness').value;
    var juiciness = document.getElementById('juiciness').value;
    var ripeness = document.getElementById('ripeness').value;
    var acidity = document.getElementById('acidity').value;
    
    var data = {
        features: [size, weight, sweetness, crunchiness, juiciness, ripeness, acidity]
    };

    fetch('/predict', {
        method: 'POST',
        body: JSON.stringify(data),
        headers:{
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        var resultDiv = document.getElementById('result');
        if (data.prediction[0] == 1) {
            resultDiv.innerHTML = 'Positive';
        } else {
            resultDiv.innerHTML = 'Negative';
        }
    })
    .catch(error => console.error('Error:', error));
}
