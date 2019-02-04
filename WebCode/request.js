//Ajax POST Method TEST
$.ajax({
    url: "http://34.73.66.149:2222/",
    dataType: "json",
    type: "POST",
    data: {
        input: [1, 4, 2, 4, 1]
    },
    success: function(resultData) {
        console.log(resultData.reply);
        var content = document.getElementById("content");
        content.innerHTML = resultData.reply;
    }
});