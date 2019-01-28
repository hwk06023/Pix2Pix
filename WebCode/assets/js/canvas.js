let pos = {
    drawable: false,
    x : -1,
    y : -1
};

let canvas, ctx;

window.onload = function() {
    canvas = document.getElementById("canvas");
    ctx = canvas.getContext("2d");

    canvas.addEventListener("mousedown", listener);
    canvas.addEventListener("mousemove", listener);
    canvas.addEventListener("mouseup", listener);
    canvas.addEventListener("mouseout", listener);
};

function getPosition(event) {
    let x = event.pageX - canvas.offsetLeft;
    let y = event.pageY - canvas.offsetTop;
    return {X:x, Y:y};
}

function initDraw() {
    ctx.beginPath();
    pos.drawable = true;
    let coors = getPosition(event);

    pos.X = coors.X;
    pos.Y = coors.Y;
    ctx.moveTo(pos.X, pos.Y);
}

function draw(event) {
    let coors = getPosition(event);
    ctx.lineTo(coors.X, coors.Y);


    pos.X = coors.X;
    pos.Y = coors.Y;
    ctx.stroke();
}


function finishDraw() {
    pos.drawable = false;
    pos.X = -1;
    pos.Y = -1;
    
}

function listener(event) {
    switch (event.type) {
        case "mousedown":
            initDraw(event);
            break;

        case "mousemove":
            if (pos.drawable)
                draw(event);
            break;

        case "mouseout":

        case "mouseup":
            finishDraw();
            break;
    }

}

