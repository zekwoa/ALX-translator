function record(){
    var recognition = new webkitSpeechRecognition();
    recognition.lang='en-IN';

    recognition.onresult = function(event){
        console.log(event)
        document.getElementById('text').value = event.results[0][0].transcript;
    }
    recognition.start();
 }