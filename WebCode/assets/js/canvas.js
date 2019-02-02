let initPos = {
    drawable: false,
    x: -1,
    y: -1
};

let curPos = {
    drawable: false,
    x: -1,
    y: -1
};

let canvas, ctx;
var HisArray = new Array();
var Step = -1;

window.onload = function() {
    canvas = document.getElementById("canvas");
    ctx = canvas.getContext("2d");

    Push();
    Push();
    canvas.addEventListener("mousedown", listener);
    canvas.addEventListener("mousemove", listener);
    canvas.addEventListener("mouseup", listener);
    canvas.addEventListener("mouseout", listener);
};

// 현재 상태를 history 배열에 넣어서 저장
function Push() {
    Step++;
    console.log("push " + Step + " : " + HisArray.length);
    if (Step < HisArray.length) { HisArray.length = Step; }
    console.log("push " + Step + " : " + HisArray.length);
    HisArray.push(canvas.toDataURL()); //그리기 상태 저장
    console.log("push " + Step + " : " + HisArray.length);
}

// 
function Undo() {
    if (Step > 0) {
        Step--;
        var canvasPic = new Image();
        var prevPic = new Image();
        canvasPic.src = HisArray[Step];
        console.log("undo to " + Step);
        ctx.clearRect(0, 0, 256, 256);
        ctx.drawImage(canvasPic, 0, 0);

        canvasPic.onload = function() { ctx.drawImage(canvasPic, 0, 0); }
    }
}

function Redo() {
    if (Step < HisArray.length - 1) {
        Step++;
        var canvasPic = new Image();
        canvasPic.src = HisArray[Step];
        console.log("redo to " + Step);
        canvasPic.onload = function() { ctx.drawImage(canvasPic, 0, 0); }
    }
}


function getPosition(event) {
    let x = event.pageX - canvas.offsetLeft;
    let y = event.pageY - canvas.offsetTop;
    return { X: x, Y: y };
}

function initDraw() {
    ctx.beginPath();
    curPos.drawable = true;
    let coors = getPosition(event);

    curPos.X = coors.X;
    curPos.Y = coors.Y;
    initPos.X = coors.X;
    initPos.Y = coors.Y;
    ctx.moveTo(curPos.X, curPos.Y);
    return curPos;
}

function draw(event) {
    let coors = getPosition(event);

    curPos.X = coors.X;
    curPos.Y = coors.Y;
    console.log("current Step : " + Step)
    Undo();
    ctx.fillRect(initPos.X, initPos.Y, curPos.X - initPos.X, curPos.Y - initPos.Y);
    Push();
}


function finishDraw() {
    curPos.drawable = false;
    curPos.X = -1;
    curPos.Y = -1;

}

function listener(event) {
    switch (event.type) {
        case "mousedown":
            initDraw(event);
            break;

        case "mousemove":
            if (curPos.drawable)
                draw(event);
            break;

        case "mouseout":
            break;

        case "mouseup":
            Push();
            finishDraw();
            break;
    }

}