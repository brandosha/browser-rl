importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.4.0/dist/tf.min.js")
if (false) { var tf = require("@tensorflow/tfjs") } // for code completion purposes

/**
 * @typedef StepData
 * @property {tf.NamedTensorMap} gradients
 * @property {number} reward
 * @property {number} action
 * @property {number} actionProbablity
 * @property {tf.Tensor} state
 */

/**
 * @class
 * @property {tf.LayersModel} model
 * @property {tf.Optimizer} optimizer
 * @property {StepData[]} steps
 */
class PolicyNetwork {
    /**
     * @param {tf.LayersModel} model 
     * @param {tf.Optimizer} optimizer 
     */
    constructor(model, optimizer) {
        this.model = model
        this.optimizer = optimizer

        this.modelIsRecurrent = model.layers.some(layer => layer instanceof tf.RNN)

        /**
         * @type {StepData[]}
         */
        this.steps = []
    }

    /**
     * @callback ActionRewardFn
     * @param {number} action 
     * @param {tf.Tensor} state 
     * @returns {number} reward 
     */

    /**
     * @param {tf.Tensor} state 
     * @param {ActionRewardFn} rewardFn 
     */
    step(state, rewardFn) {
        var action = 0
        var actionProbablity = 0
        var gradients = tf.tidy(() => {
            var stateBatch = state.reshape([1].concat(state.shape))

            var gradients = this.optimizer.computeGradients(() => {
                var prediction = this.model.predict(stateBatch)
                var out = prediction.squeeze()

                if (this.modelIsRecurrent) {
                    var outputs = out.shape[0]
                    out = out.slice(outputs - 1, 1)
                }

                var probabilities = out.dataSync()
                var rand = Math.random()
                probabilities.some((prob, index) => {
                    action = index
                    actionProbablity = prob
                    return (rand -= prob) <= 0
                })

                if (out.shape[0] > 1) {
                    var actionOneHot = tf.oneHot(action, probabilities.length)
                } else {
                    var actionOneHot = tf.tensor1d([1])
                }
                
                return out.mul(actionOneHot).sum()
            }).grads

            return gradients
        })

        this.steps.push({
            state: state,
            action: action,
            actionProbablity: actionProbablity,
            reward: tf.tidy(() => rewardFn(action, state)),
            gradients: gradients
        })
    }

    /**
     * @param {number} discountRate 
     */
    discountRewards(discountRate) {
        var nextStepReward = 0
        for (let i = this.steps.length - 1; i > 0; i--) {
            var discountedReward = this.steps[i].reward + nextStepReward * discountRate
            this.steps[i].reward = discountedReward
            nextStepReward = discountedReward
        }
    }

    /**
     * @callback GradScaleFn
     * @param {StepData} step
     * @returns {tf.NamedTensorMap}
     */

    /**
     * @param {GradScaleFn} scaleFn 
     */
    scaleAndApplyGradients(scaleFn) {
        tf.tidy(() => {
            var allGradients = { }
            this.steps.forEach((step, i) => {
                var scaledGradients = scaleFn(step)
                tf.dispose(this.steps[i].gradients)
                this.steps[i].state.dispose()

                for (const name in scaledGradients) {
                    if (name in allGradients) {
                        allGradients[name].push(scaledGradients[name])
                    } else {
                        allGradients[name] = [scaledGradients[name]]
                    }
                }
            })

            var avgGradients = { }
            for (const name in allGradients) {
                var stackedGradients = tf.stack(allGradients[name])
                avgGradients[name] = stackedGradients.mean(0).mul(-1)
            }

            this.optimizer.applyGradients(avgGradients)
            this.steps = []
        })
    }
}