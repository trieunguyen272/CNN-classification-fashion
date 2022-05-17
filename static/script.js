var file = document.getElementById("image"),
    preview = document.getElementById("preview");

file.onchange = function() {
    if (file.files.length > 0) {
        document.getElementById('file-name').innerHTML = file.files[0].name;
    }
};

file.addEventListener("change", function() {
    changeImage(this);
});
$(function() {
    $("#classify-button").hide();

});

function changeImage(input) {
    var reader;

    if (input.files && input.files[0]) {
        reader = new FileReader();

        reader.onload = function(e) {
            preview.setAttribute('src', e.target.result);
        }

        reader.readAsDataURL(input.files[0]);
        $("#classify-button").show();
    }
}