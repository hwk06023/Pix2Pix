var initPos = {
    drawable: false,
    x: -1,
    y: -1
};

var curPos = {
    drawable: false,
    x: -1,
    y: -1
};

var colorDict = {
    "background": "#0006D9",
    "wall": "#0D3DFB",
    "door": "#A50000",
    "window": "#0075FF",
    "window_sill": "#68F898",
    "window_head": "#1DFFDD",
    "shutter": "#EEED28",
    "balcony": "#B8FF38",
    "trim": "#FF9204",
    "cornice": "#FF4401",
    "column": "#F60001",
    "entrance": "#00C9FF"
}

var canvas, ctx, body, output, otx;
var HisArray = new Array();
var Step = -1;

window.onload = function() {
    canvas = document.getElementById("canvas");
    output = document.getElementById("output");
    body = document.getElementById("body");
    ctx = canvas.getContext("2d");
    otx = output.getContext("2d");


    //canvas.style.backgroundColor = colorDict["wall"];
    ctx.fillStyle = colorDict["wall"];
    ctx.fillRect(0, 0, 256, 256);
    Push();
    Push();
    Push();

    canvas.addEventListener("mousedown", listener);
    canvas.addEventListener("mousemove", listener);
    canvas.addEventListener("mouseover", listener);
    canvas.addEventListener("mouseup", listener);
    canvas.addEventListener("mouseout", listener);
};

// 현재 상태를 history 배열에 넣어서 저장
function Push() {
    Step++;
    //console.log("push " + Step + " : " + HisArray.length);
    if (Step < HisArray.length) { HisArray.length = Step; }
    //console.log("push " + Step + " : " + HisArray.length);
    HisArray.push(canvas.toDataURL()); //그리기 상태 저장
    //console.log("push " + Step + " : " + HisArray.length);
}

// 
function Undo(callby) {
    if (Step > 0) {
        Step--;
        var canvasPic = new Image();
        var prevPic = new Image();
        canvasPic.src = HisArray[Step];
        if (callby == "btn") {
            canvasPic.src = HisArray[Step - 1];
        }
        //console.log("undo to " + Step);
        ctx.clearRect(0, 0, 256, 256);
        ctx.drawImage(canvasPic, 0, 0);
        otx.drawImage(canvasPic, 0, 0)

        canvasPic.onload = function() { ctx.drawImage(canvasPic, 0, 0); }
    }
}

function Redo() {
    if (Step < HisArray.length - 1) {
        Step++;
        var canvasPic = new Image();
        canvasPic.src = HisArray[Step - 1];
        //console.log("redo to " + Step);
        canvasPic.onload = function() { ctx.drawImage(canvasPic, 0, 0); }
    }
}


function getPosition(event) {
    var x = event.pageX - canvas.offsetLeft;
    var y = event.pageY - canvas.offsetTop;
    return { X: x, Y: y };
}

function initDraw() {
    ctx.beginPath();
    curPos.drawable = true;
    var coors = getPosition(event);

    curPos.X = coors.X;
    curPos.Y = coors.Y;
    initPos.X = coors.X;
    initPos.Y = coors.Y;
    ctx.moveTo(curPos.X, curPos.Y);
    return curPos;
}

function draw(event) {
    var coors = getPosition(event);

    curPos.X = coors.X;
    curPos.Y = coors.Y;
    //console.log("current Step : " + Step)
    Undo("drawing");
    ctx.fillRect(initPos.X, initPos.Y, curPos.X - initPos.X, curPos.Y - initPos.Y);
    Push("drawing");
}


function finishDraw() {

    curPos.drawable = false;
    curPos.X = -1;
    curPos.Y = -1;

}

function getColor() {
    var colors = document.getElementsByName("color");
    var element;
    for (var i = 0; i < colors.length; i++) {
        if (colors[i].checked) {
            element = colors[i].value;
            break;
        }
    }
    return colorDict[element];
}

function listener(event) {
    switch (event.type) {
        case "mousedown":
            ctx.fillStyle = getColor();
            initDraw(event);
            break;

        case "mousemove":
            if (curPos.drawable)
                draw(event);
            break;

        case "mouseover":
            body.style.cursor = "crosshair";
            break;

        case "mouseout":
            body.style.cursor = "default";
            break;

        case "mouseup":
            ctx.fillRect(initPos.X, initPos.Y, curPos.X - initPos.X, curPos.Y - initPos.Y);
            Push();
            finishDraw();
            break;
    }

}

function Print() {
    var a = document.getElementsByName("color");
    var b;
    for (var i = 0; i < a.length; i++) {
        if (a[i].checked) {
            b = a[i].value;
        }
    }
    console.log(b);
}