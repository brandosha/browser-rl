var map = Array(100).fill(undefined).map(_ => Array(100).fill(0))
var trail = Array(100).fill(undefined).map(_ => Array(100).fill(0))

var player = {
    x: 0,
    y: 0
}

var currentKey = null

var canvas = document.getElementById("game")
canvas.height = map.length
canvas.width = map[0].length
canvas.style = "height: 50%; image-rendering: pixelated;"

var ctx = canvas.getContext('2d')
ctx.imageSmoothingEnabled = false
function draw() {
    Array(map.length).fill(undefined).forEach((_, y) => {
        Array(map[0].length).fill(undefined).forEach((_, x) => {
            if (map[y][x]) {
                ctx.fillStyle = "rgb(100, 100, 100)"
            } else if (player.y === y && player.x === x) {
                ctx.fillStyle = "rgb(200, 0, 0)"
            } else {
                ctx.fillStyle = "rgb(0, 0, 0)"
            }

            ctx.fillRect(x, y, 1, 1)
        })
    })

    if (currentKey === "ArrowUp")         movePlayer(0, -1)
    else if (currentKey === "ArrowDown")  movePlayer(0, 1)
    else if (currentKey === "ArrowRight") movePlayer(1, 0)
    else if (currentKey === "ArrowLeft")  movePlayer(-1, 0)

    setTimeout(draw, 1000 / 60)
}
draw()

function movePlayer(x, y) {
    var newX = player.x + x
    var newY = player.y + y

    if (newX >= map[0].length || newY >= map.length || newX < 0 || newY < 0) { return }
    if (map[newY][newX]) { return }

    map[player.y][player.x] = 1

    player.x = newX
    player.y = newY
}

document.addEventListener('keydown', (e) => { currentKey = e.code });