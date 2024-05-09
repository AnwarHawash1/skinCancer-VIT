$('#fileup').change(function(){
    // Get the file extension and set an array of valid extensions
    var res = $('#fileup').val();
    var arr = res.split("\\");
    var filename = arr.slice(-1)[0];
    var filextension = filename.split(".");
    var filext = "." + filextension.slice(-1)[0];
    var valid = [".jpg", ".png", ".jpeg", ".bmp"];

    // If the file is not valid, show the error icon, the red alert, and hide the submit button
    if (valid.indexOf(filext.toLowerCase()) == -1) {
        $(".imgupload").hide("slow");
        $(".imgupload.ok").hide("slow");
        $(".imgupload.stop").show("slow");

        $('#namefile').css({"color": "red", "font-weight": 700});
        $('#namefile').html("File " + filename + " is not a pic!");

        $("#submitbtn").hide();
        $("#fakebtn").show();
    } else {
        // If the file is valid, show the green alert and the valid submit button
        $(".imgupload").hide("slow");
        $(".imgupload.stop").hide("slow");
        $(".imgupload.ok").show("slow");

        $('#namefile').css({"color": "green", "font-weight": 700});
        $('#namefile').html(filename);

        $("#submitbtn").show();
        $("#fakebtn").hide();
    }
});
