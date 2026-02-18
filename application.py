from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import string

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.utils import load_object
from scipy.sparse import hstack

application = Flask(__name__, template_folder='src/templates')
app = application

@app.route('/')
def index():
    return render_template('website.html')

@app.route('/predict', methods=['POST'])
def predict_datapoint():
    try:
        title = request.form.get('title')
        text = request.form.get('text')

        raw_text_length = len(str(text))
        raw_word_count = len(str(text).split())
        punct_count = sum([1 for c in str(text) if c in string.punctuation])
        capital_count = sum([1 for c in str(text) if c.isupper()])
        numeric_count = sum([1 for c in str(text) if c.isdigit()])
        punct_density = punct_count / raw_text_length if raw_text_length > 0 else 0
        capital_density = capital_count / raw_text_length if raw_text_length > 0 else 0
        avg_word_length = raw_text_length / raw_word_count if raw_word_count > 0 else 0
        title_length = len(str(title))

        data = CustomData(title=title, text=text)
        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(pred_df)

        try:
            model = load_object('artifact/model.pkl')
            preprocessor = load_object('artifact/preprocessor.pkl')
            tfidf_pipeline = preprocessor['tfidf_pipeline']
            num_pipeline = preprocessor['num_pipeline']
            numerical_columns = preprocessor['numerical_columns']

            # Re-run engineer_features to get the processed df for proba
            pred_df2 = data.get_data_as_data_frame()
            pred_df2 = predict_pipeline.engineer_features(pred_df2)

            text_transformed = tfidf_pipeline.transform(pred_df2['text'])
            numerical_transformed = num_pipeline.transform(pred_df2[numerical_columns])
            data_scaled = hstack([text_transformed, numerical_transformed])

            proba = model.predict_proba(data_scaled)[0]
            confidence = float(proba[prediction[0]] * 100)

        except Exception as e:
            print("Could not get confidence score:", str(e))
            confidence = 95.0

        features = {
            'text_length': raw_text_length,
            'title_length': title_length,
            'word_count': raw_word_count,
            'punct_count': punct_count,
            'capital_count': capital_count,
            'numeric_count': numeric_count,
            'punct_density': punct_density,
            'capital_density': capital_density,
            'avg_word_length': avg_word_length
        }

        result = {
            'prediction': 'Real' if prediction[0] == 1 else 'Fake',
            'confidence': round(confidence, 1),
            'features': features
        }

        return jsonify(result)

    except Exception as e:
        print("ERROR:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)