$(document).ready(function () {
    // init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result-text').hide();

    // upload the image for preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(500);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-caption').show();
        $('#result-text').text('');
        $('#result-text').hide();
        readURL(this);
    });

    // generate caption
    $('#btn-caption').click(function () {
        let form_data = new FormData($('#upload-file')[0]);

        // show loading animation
        $(this).hide();
        $('.loader').show();

        // caption image by calling api /caption
        $.ajax({
            type: 'POST',
            url: '/caption',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            dataType: "json",
            success: function (data) {
                let task_id = data.task_id;
                if (task_id) {
                    let intervalID = setInterval(function() {
                            $.ajax({
                                type: 'GET',
                                dataType: "json",
                                url: `/caption/${task_id}`,
                                async: true,
                                success: function (data) {
                                    if (data.ready) {
                                        clearInterval(intervalID);
                                        $('.loader').hide();
                                        $('#result-text').fadeIn(500);
                                        $('#result-text').text('Proposed caption: ' + data.result);                                        
                                    }
                                }
                            });
                        },
                        200
                    )
                }
            }
        });
    });
});
