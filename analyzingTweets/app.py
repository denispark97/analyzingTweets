from flask import Flask, request, render_template
import model
text_classifier, vectorizer, custom_neg_specific, labels = model.run_model()
app = Flask(__name__, template_folder = 'template')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def predict():
     #obtain all form values and place them in an array, convert into integers
    user_input = request.form['input']
    prediction = model.predict(user_input, text_classifier, vectorizer, custom_neg_specific)
    rcmd_top_likes, top_likes = model.recmd_tweets(prediction, labels)
    likes = []
    rts = []
    txts=[]
    for i in rcmd_top_likes['like_count']:
        likes.append(i)
    for i in rcmd_top_likes['retweet_count']:
        rts.append(i)
    for i in rcmd_top_likes['text']:
        txts.append(i)
    if prediction == 1:
        user_txt = "You agree with this incident"
    else:
        user_txt = "You disagree with this incident"

    return render_template('index.html',pred=user_txt, 
    like0=likes[0],like1=likes[1],like2=likes[2],like3=likes[3],like4=likes[4],rts0=rts[0],
    rts1=rts[1],rts2=rts[2],rts3=rts[3],rts4=rts[4],txt0=txts[0],txt1=txts[1],txt2=txts[2],
    txt3=txts[3],txt4=txts[4])

if __name__ == "__main__":
    app.run(debug=True)

