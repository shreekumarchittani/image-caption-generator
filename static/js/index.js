var loadFile = function(event) {
    var image = document.getElementById('output');
    var btn = document.getElementById('upload-btn');
    image.src = URL.createObjectURL(event.target.files[0]);
    image.removeAttribute('hidden');
    btn.removeAttribute('disabled');
    btn.style.cursor = "";
};

