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

var rewardsCsv = ""

function train() {
    var totalReward = 0
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
    
    policyNetwork.discountRewards(0)
    policyNetwork.scaleAndApplyGradients((step) => {
        totalReward += step.reward

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
