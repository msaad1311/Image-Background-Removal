import * as tf from "@tensorflow/tfjs";
var openFile = async function(file) {
    var input = file.target;
    const weights = "model.json";
    const model = await tf.loadGraphModel(weights);
    console.log(model)
    var reader = new FileReader();
    reader.onload = function(){
      var dataURL = reader.result;
      var output = document.getElementById('output');
      output.src = dataURL;
    };
    reader.readAsDataURL(input.files[0]);
  };