importScripts("PolicyNetwork.js")
if (false) { var tf = require("@tensorflow/tfjs") } // for code completion purposes

var model = tf.sequential({
    layers: [
        tf.layers.inputLayer({ inputShape: [3] }),
        tf.layers.dense({ units: 3 }),
        tf.layers.softmax()
    ]
})

var optimizer = tf.train.adam(0.1)

var policyNetwork = new PolicyNetwork(model, optimizer)

var agentRewards = []
var randomRewards = []

function train() {
    var randomReward = 0

    function gameRound() {
        var start = performance.now()
        var randIndex = Math.floor(Math.random() * 3)
        var input = Array(3).fill(0)
        input[randIndex] = 1

        randomReward += input[Math.floor(Math.random() * 3)]
    
        policyNetwork.step(tf.tensor1d(input), action => input[action])
        var start = performance.now()
    }
    
    Array(50).fill(undefined).forEach(gameRound)

    var totalReward = policyNetwork.steps.map(step => step.reward).reduce((a,b) => a + b, 0)
    
    policyNetwork.standardizeRewards()
    policyNetwork.scaleAndApplyGradients((step) => {
        var scaledGradients = { }
        for (const name in step.gradients) {
            scaledGradients[name] = step.gradients[name].mul(step.reward).div(step.actionProbablity)
        }
    
        return scaledGradients
    })

    agentRewards.push(totalReward)
    randomRewards.push(randomReward)
}

Array(50).fill(undefined).forEach(train)

console.log(agentRewards.join("\n"))
console.log(randomRewards.join("\n"))
