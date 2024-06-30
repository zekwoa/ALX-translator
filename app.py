import cv2
import mediapipe as mp
import numpy as np
import pickle
import json
from flask import Flask, render_template, request, Response
import nltk
import time

# For Filtring wrords
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
wnl = nltk.stem.WordNetLemmatizer()

# From Sign To Text
# Load the saved SVM model
with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)

# Initialize the Flask app
app = Flask(__name__)

# Define the server's route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/home')
def home():
    return render_template('home.html')

# Initialize an empty list to store the prediction results
predictions = []

# Define the function to process the hand image and make predictions
def image_processed(hand_img):
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    # Flip the image in Y-axis
    img_flip = cv2.flip(img_rgb, 1)

    # Accessing MediaPipe solutions
    mp_hands = mp.solutions.hands

    # Initialize Hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7)

    # Process the image
    output = hands.process(img_flip)

    # Close the Hands object
    hands.close()

    try:
        # Extract the hand landmarks from the output
        data = output.multi_hand_landmarks[0]
        data = str(data)
        data = data.strip().split('\n')

        # Remove the unnecessary information from the landmark data
        garbage = ['landmark {', ' visibility: 0.0', ' presence: 0.0', '}']
        without_garbage = []
        for i in data:
            if i not in garbage:
                without_garbage.append(i)

        clean = []
        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])

        # Return the cleaned hand landmark data
        return(clean)
    except:
        # If no hand landmarks are detected, return an array of zeros
        return(np.zeros([1,63], dtype=int)[0])

# Define the video capture function
def generate_frames():
    # Initialize the video capture
    cap = cv2.VideoCapture(0)

    # Loop over the frames from the video stream
    while True:
        # Read a frame from the video stream
        success, frame = cap.read()

        if not success:
            break

        # Process the frame to get the hand landmarks
        hand_landmarks = image_processed(frame)

        # Use the SVM to make a prediction based on the landmarks
        prediction = svm.predict([hand_landmarks])[0]

        # Append the prediction to the predictions list
        predictions.append(prediction)

        # Draw the prediction on the frame
        cv2.putText(frame, prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode the frame as a jpeg image
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the encoded frame
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release the video capture
    cap.release()

# Define the route for the video stream
@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Define the route for getting the prediction result
@app.route('/results.json')
def results():
    # If the predictions list is not empty, return the last prediction as a JSON object
    if predictions:
        return json.dumps({'prediction': predictions[-1]})
    else:
        # If the predictions list is empty, return an empty JSON object
        return json.dumps({})
    
# From Text/Voice to Sign
# Define the route for processing the form submission
@app.route('/submit', methods=['POST'])
def submit():
    text = request.form['text']
    stop = nltk.corpus.stopwords.words('english')
    stop_words=['@','#',"http",":","is","the","are","am","a","it","was","were","an",",",".","?","!",";","/"]
    for i in stop_words:
        stop.append(i)

    #processing the text using bag of wor
    tokenized_text = nltk.tokenize.word_tokenize(text)
    lemmed = [wnl.lemmatize(word) for word in tokenized_text]
    processed=[]
    for i in lemmed :
        if i == "i" or i == "I":
            processed.append("me")
        elif i not in stop:
            i=i.lower()
            processed.append((i))
    print("Keywords:",processed)

    #Showing animation of the keywords.
    assets_list=['0.mp4', '1.mp4', '2.mp4', '3.mp4', '4.mp4', '5.mp4','6.mp4', '7.mp4', '8.mp4', '9.mp4', 'a.mp4', 'after.mp4',
                'again.mp4', 'against.mp4', 'age.mp4', 'all.mp4', 'alone.mp4','also.mp4', 'and.mp4', 'ask.mp4', 'at.mp4', 'b.mp4', 'be.mp4',
                'beautiful.mp4', 'before.mp4', 'best.mp4', 'better.mp4', 'busy.mp4', 'but.mp4', 'bye.mp4', 'c.mp4', 'can.mp4', 'cannot.mp4',
                'change.mp4', 'college.mp4', 'come.mp4', 'computer.mp4', 'd.mp4', 'day.mp4', 'distance.mp4', 'do not.mp4', 'do.mp4', 'does not.mp4',
                'e.mp4', 'eat.mp4', 'engineer.mp4', 'f.mp4', 'fight.mp4', 'finish.mp4', 'from.mp4', 'g.mp4', 'glitter.mp4', 'go.mp4', 'god.mp4',
                'gold.mp4', 'good.mp4', 'great.mp4', 'h.mp4', 'hand.mp4', 'hands.mp4', 'happy.mp4', 'hello.mp4', 'help.mp4', 'her.mp4', 'here.mp4',
                'his.mp4', 'home.mp4', 'homepage.mp4', 'how.mp4', 'i.mp4', 'invent.mp4', 'it.mp4', 'j.mp4', 'k.mp4', 'keep.mp4', 'l.mp4', 'language.mp4', 'laugh.mp4',
                'learn.mp4', 'm.mp4', 'me.mp4', 'mic3.png', 'more.mp4', 'my.mp4', 'n.mp4', 'name.mp4', 'next.mp4', 'not.mp4', 'now.mp4', 'o.mp4', 'of.mp4', 'on.mp4',
                'our.mp4', 'out.mp4', 'p.mp4', 'pretty.mp4', 'q.mp4', 'r.mp4', 'right.mp4', 's.mp4', 'sad.mp4', 'safe.mp4', 'see.mp4', 'self.mp4', 'sign.mp4', 'sing.mp4', 
                'so.mp4', 'sound.mp4', 'stay.mp4', 'study.mp4', 't.mp4', 'talk.mp4', 'television.mp4', 'thank you.mp4', 'thank.mp4', 'that.mp4', 'they.mp4', 'this.mp4', 'those.mp4', 
                'time.mp4', 'to.mp4', 'type.mp4', 'u.mp4', 'us.mp4', 'v.mp4', 'w.mp4', 'walk.mp4', 'wash.mp4', 'way.mp4', 'we.mp4', 'welcome.mp4', 'what.mp4', 'when.mp4', 'where.mp4', 
                'which.mp4', 'who.mp4', 'whole.mp4', 'whose.mp4', 'why.mp4', 'will.mp4', 'with.mp4', 'without.mp4', 'words.mp4', 'work.mp4', 'world.mp4', 'wrong.mp4', 'x.mp4', 'y.mp4',
                'you.mp4', 'your.mp4', 'yourself.mp4', 'z.mp4']
    tokens_sign_lan=[]

    for word in processed:
        string = str(word+".mp4")
        if string in assets_list:
            tokens_sign_lan.append(str("assets/"+string))
        else:
            for j in word:
                tokens_sign_lan.append(str("assets/"+j+".mp4"))

    def generate_frames(video_file):
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise ValueError("Error File Not Found")
        while True:
            label = video_file.replace("assets/","").replace(".mp4","")
            fps= int(cap.get(cv2.CAP_PROP_FPS))
            ret, frame = cap.read()
            if not ret:
                break
            time.sleep(1/fps)
            frame = cv2.putText(frame, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()

    # Concatenate the video frames from all the video files
    def generate_all_frames():
        for video_file in tokens_sign_lan:
            label = video_file.replace("assets/","").replace(".mp4","")
            yield from generate_frames(video_file)

    return Response(generate_all_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Server
if __name__ == '__main__':
    app.run(debug=True)