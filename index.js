var path = require('path');
var fs = require('fs');
var os = require('os');
var express = require('express');
var app = express();
//var busboy = require('busboy');
var busboy = require('connect-busboy');
var path = require('path');

const tf = require('@tensorflow/tfjs-node');
const labels = require('./model/ABCDX_iphone/assets/labels.json');

let objectDetectionModel;

function indexOfMax(arr) {
    if (arr.length === 0) {
        return -1;
    }

    var max = arr[0];
    var maxIndex = 0;

    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }
    return maxIndex;
}

async function loadModel() {
  // Warm up the model
  if (!objectDetectionModel) {
    // Load the TensorFlow SavedModel through tfjs-node API. You can find more
    // details in the API documentation:
    // https://js.tensorflow.org/api_node/1.3.1/#node.loadSavedModel
	//objectDetectionModel = tf.node.loadSavedModel(
    objectDetectionModel = await tf.node.loadSavedModel(	
      './model/ABCDX_iphone', ['serve'], 'serving_default');
    //new_object_detection_1
  }
  const tempTensor = tf.zeros([1, 2, 2, 3]).toInt();
  objectDetectionModel.predict(tempTensor);
}

app.get('/', async function (req, res) {
	res.sendFile(path.join(__dirname + '/index.html'));
//    res.send('<html><head></head><body>\
//               <form method="POST" action="/save" enctype="multipart/form-data">\
//                <input type="text" name="textfield"><br />\
//                <input type="file" name="filefield"><br />\
//                <input type="submit">\
//              </form>\
//            </body></html>');
			
  //res.end();
});


// accept POST request on the homepage { immediate: true }
app.post('/predict', busboy({ immediate: true }), async (req, res, next) => {
  console.log('in predict')
  if (!req.busboy) throw new Error('file binary data cannot be null');
  
  let fileData = null;
  let token = null;
  req.files = { file: [] };
  req.busboy.on('file', (fieldName, file, filename, encoding, mimetype) => {
	console.log('in file')
    file.on('data', (data) => {
      if (fileData === null) {
        fileData = data;
      } else {
        fileData = Buffer.concat([fileData, data]);
      }
    });
	file.on('end', () => {
      const file_object = {
        fieldName,
        'originalname': filename,
        encoding,
        mimetype,
        buffer: fileData
      };
	  console.log(encoding, mimetype, filename);
      req.files.file.push(file_object)
	});
  });
  req.busboy.on('field', (fieldName, value) => {
	console.log('in field', fieldName)
    if (fieldName === 'image') {
      img = value;
	  console.log(img)
    }
  });
  req.busboy.on('finish', async () =>{
    if (!fileData) next(new Error('file binary data cannot be null'));
    //	if (!token) next(new Error('No security token was passed'));
    //TODO: use your parsed parameters to complete the request
  //});
  //busboy.on('finish', async () => {
    const buf = req.files.file[0].buffer;
    const uint8array = new Uint8Array(buf);

	console.log(buf.shape)
    // Decode the image into a tensor.
    const imageTensor = await tf.node.decodeImage(uint8array);
    const input = imageTensor.expandDims(0);

    // Feed the image tensor into the model for inference.
    const startTime = tf.util.now();
    let outputTensor = objectDetectionModel.predict({ 'input_tensor': input });

    // Parse the model output to get meaningful result(get detection class and
    // object location).
    const scores = await outputTensor['detection_scores'].arraySync();
    const boxes = await outputTensor['detection_boxes'].arraySync();
    const names = await outputTensor['detection_classes'].arraySync();
	console.log(names)
    const endTime = tf.util.now();
    outputTensor['detection_scores'].dispose();
    outputTensor['detection_boxes'].dispose();
    outputTensor['detection_classes'].dispose();
    outputTensor['num_detections'].dispose();
    const detectedBoxes = [];
    const detectedNames = [];
    const detectedScores = [];
    const detectedObjArray = [];
    const detectedHeight = [];
    for (let i = 0; i < scores[0].length; i++) {
      if (scores[0][i] > 0.1) {
        detectedBoxes.push(boxes[0][i]);
        detectedNames.push(labels[names[0][i]]);
        detectedScores.push(scores[0][i]);
        detectedObjArray.push({name:labels[names[0][i]], score:scores[0][i], box:boxes[0][i], y_loc:boxes[0][i][0]});
        detectedHeight.push(boxes[0][i][2] - boxes[0][i][0]);
      }
    }
    const average = array => array.reduce((a, b) => a + b) / array.length;
    console.log(average(detectedHeight));
    const average_height = average(detectedHeight) / 2;   
    detectedObjArray.sort((a, b) => a.y_loc - b.y_loc);
    const cleanObjArray = [];
    let i = 0; 
    while(i < detectedObjArray.length) {
        console.log('i: %d, %f', i, detectedObjArray[i].y_loc)
        const overlap = [];
        let j = i        
        while((j < detectedObjArray.length)&&(detectedObjArray[j].y_loc - detectedObjArray[i].y_loc < average_height)){
            console.log('j: %d, %f, %f', j, detectedObjArray[j].y_loc, detectedObjArray[j].score);
            overlap.push(detectedObjArray[j].score);
            j++;
        }
        const idx = indexOfMax(overlap)
        cleanObjArray.push(detectedObjArray[i+idx])
        i = j;
    }
    //const answers_901R =  ['B','C','A','C','D','C','A','B','D','C','B','B','A','B','C','A','C','B','C','D','C','B','A','D','B','D','C','A','C','D','C','B','D','B','C','B','C','B','A'];    
    const answers_905R =  ['A','D','B','A','D','B','D','C','B','C','B','C','B','D','D','A','B','C','C','B','D','B','A','D','B','D','A','B','C','C','B','D','A','B','B','A','D','A','C','B'];
    let score = 0;
    for (let i = 0; i < cleanObjArray.length; i++) {
        if(cleanObjArray[i].name === answers_905R[i]){
            score++;
        }
    }
    
    const to_scaled = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:210,12:211,12:213,12:214,15:215,16:216,17:217,18:219,19:220,20:221,21:222,22:223,23:224,24:225,25:227,26:228,27:229,28:230,29:232,30:233,31:235,32:236,33:237,34:238,35:238,36:238,37:238,38:238,39:238,40:238}
    console.log("score: %d, %d",score,to_scaled[score]);
    res.send({
      //boxes: detectedBoxes,
      //names: detectedNames,
      //scores: detectedScores,
      annotated_boxes: cleanObjArray,
      inferenceTime: endTime - startTime,
      score: to_scaled[score]
    });
  });
});

var server = app.listen(5000, function () {

  var host = server.address().address
  var port = server.address().port
  loadModel();
  console.log('Example app listening at http://%s:%s', host, port)

});