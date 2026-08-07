# Mini Project Report

## Title

InternGuard: AI-Based Fake Internship Offer Detection System

## Objective

To build a machine learning system that helps students identify suspicious internship offers by analyzing offer text and detecting common scam signals.

## Existing Problem

Many students receive internship offers from unofficial sources. These offers may ask for registration fees, promise guaranteed jobs, or use urgent language. Students need a quick first-level screening tool before trusting such offers.

## Proposed System

The proposed system accepts internship offer text from the user and classifies it as genuine or suspicious. It also displays confidence and warning signals so the result is explainable.

## Algorithm Used

The project uses TF-IDF for text feature extraction and Logistic Regression for classification.

## Modules

- Dataset module: stores labeled internship offer examples
- Training module: trains the text classification model
- Prediction module: predicts the category of new input
- Explainability module: detects suspicious keywords and patterns
- UI module: provides a Streamlit web interface

## Advantages

- Simple and fast to use
- Useful for students
- Easy to explain in a project review
- Can be completed in one day
- Expandable with real datasets and advanced NLP models

## Conclusion

InternGuard demonstrates how AI/ML can solve a practical student safety problem. The system is not a final legal or hiring verification tool, but it provides a helpful early warning mechanism for internship scams.
