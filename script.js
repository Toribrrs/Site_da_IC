// Função para redirecionar para a página desejada
function redirecionarParaOutraPagina(destino) {
        window.location.href = destino;
    }
    
    // Adicione ouvintes de evento para os botões
    document.getElementById("liquor").addEventListener("click", function () {
        redirecionarParaOutraPagina("liquor.html");
    });
    
    document.getElementById("plasma").addEventListener("click", function () {
        redirecionarParaOutraPagina("plasma.html");
    });
    
    document.getElementById("liquorplusplasma").addEventListener("click", function () {
        redirecionarParaOutraPagina("liquorplusplasma.html");
    });

function calculatePrediction() {
    const proteina1 = parseFloat(document.getElementById("proteina1").value);
    const proteina2 = parseFloat(document.getElementById("proteina2").value);
    const proteina3 = parseFloat(document.getElementById("proteina3").value);
    
    const data = {
        values: [proteina1, proteina2, proteina3]
    };
    
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        const predictionResult = data.prediction;
        document.getElementById("predictionText").textContent = "A probabilidade do paciente desenvolver Alzheimer é " + predictionResult + "%";
    });
}
    
    // Adicione um ouvinte de evento ao botão de cálculo
document.getElementById("calcular").addEventListener("click", function () {
        // Execute a função para calcular a previsão
    calculatePrediction();
    
    // Redirecione para a página results.html após calcular
    redirecionarParaOutraPagina("results.html");
});