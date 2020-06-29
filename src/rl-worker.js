importScripts("PolicyNetwork.js")
if (false) { 
    var PolicyNetwork = require("./PolicyNetwork.js")
    var tf = require("@tensorflow/tfjs")
} // for code completion purposes

var imageInput = tf.input({ shape: [100, 100, 1] })
var posInput = tf.input({ shape: [2] })

var conv1 = tf.layers.conv2d({
    filters: 3,
    kernelSize: 5
}).apply(imageInput)
var max1 = tf.layers.maxPooling2d({
    poolSize: 5
}).apply(conv1)
var conv2 = tf.layers.conv2d({
    filters: 3,
    kernelSize: 5
}).apply(max1)
var max2 = tf.layers.maxPooling2d({
    poolSize: 5
}).apply(conv2)
var flatten = tf.layers.flatten().apply(max2)

var posDense = tf.layers.dense({
    units: 5
}).apply(posInput)
var combine = tf.layers.concatenate().apply([flatten, posDense])
var dense1 = tf.layers.dense({
    units: 32
}).apply(combine)
var dense2 = tf.layers.dense({
    units: 32
}).apply(dense1)
var dense3 = tf.layers.dense({
    units: 32
}).apply(dense2)
var dense4 = tf.layers.dense({
    units: 4
}).apply(dense3)
var output = tf.layers.softmax().apply(dense4)

var model = tf.model({
    inputs: [imageInput, posInput],
    outputs: output
})
var optimizer = tf.train.adam(0.1)

var policyNetwork = new PolicyNetwork(model, optimizer)

var map = Array(100).fill(undefined).map(_ => Array(100).fill(0))
var player = {
    x: 50,
    y: 50
}

function goodMove(x, y) {
    var newX = player.x + x
    var newY = player.y + y

    if (newX >= map[0].length || newY >= map.length || newX < 0 || newY < 0) { return false }
    if (map[newY][newX]) { return false }

    return true
}

var gameOver = false

onmessage = message => {
    if (gameOver) return

    var data = message.data

    // console.log(data)

    var trails = tf.tensor2d(data.trail).reshape([100, 100, 1])
    var pos = tf.tensor1d([data.player.x, data.player.y])

    // console.log(posInput.shape)
    // console.log(pos.arraySync(), [data.player.x, data.player.y], trails.arraySync(), data.trail)
    // console.log(model.execute([trails, pos]))//.call([trails, pos]))
    

    var actions = [
        { x: 0, y: 1 },
        { x: 0, y: -1 },
        { x: 1, y: 0 },
        { x: -1, y: 0 }
    ]
    
    policyNetwork.step([trails, pos], action => {
        if (goodMove(actions[action].x, actions[action].y)) {
            return 1
        } else {
            gameOver = true
            onGameOver()
            return -10
        }
    })
    // console.log(message.data)

    var rand1 = Math.floor(Math.random() * 2)
    var rand2 = Math.floor(Math.random() * 2)

    var randomX = rand1 ? 0 : (rand2 ? -1 : 1)
    var randomY = rand1 ? (rand2 ? -1 : 1) : 0

    // console.log(randomX, randomY)

    postMessage({
        move: {
            x: randomX,
            y: randomY
        }
    })
}

function onGameOver() {
    policyNetwork.standardizeRewards()
    policyNetwork.discountRewards(0.75)
    policyNetwork.scaleAndApplyGradients(PolicyNetwork.vanilla)
}

/*var model = tf.sequential({
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
        var randIndex = Math.floor(Math.random() * 3)
        var input = Array(3).fill(0)
        input[randIndex] = 1

        randomReward += input[Math.floor(Math.random() * 3)]
    
        policyNetwork.step(tf.tensor1d(input), action => input[action])
    }
    
    Array(50).fill(undefined).forEach(gameRound)

    var totalReward = policyNetwork.steps.map(step => step.reward).reduce((a,b) => a + b, 0)

    policyNetwork.standardizeRewards()
    policyNetwork.scaleAndApplyGradients((step) => {
        return PolicyNetwork.mapGradients(step.gradients, (grad) => {
            return grad.mul(step.reward).div(step.actionProbablity)
        })
    })

    agentRewards.push(totalReward)
    randomRewards.push(randomReward)
}

Array(50).fill(undefined).forEach(train)

console.log(agentRewards.join("\n"))
console.log(randomRewards.join("\n"))
*/