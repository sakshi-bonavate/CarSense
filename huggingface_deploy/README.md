---
title: CarSense - Car Purchase Predictor
emoji: 🚗
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🚗 CarSense — Car Purchase Predictor

An AI-powered web app that predicts whether a user should buy a car based on their **gender**, **age**, and **annual salary**.

## Model
- **Algorithm:** Random Forest Classifier (200 estimators)
- **Accuracy:** 90.5% on held-out test data
- **Training data:** 1,000 real car purchase records
- **Key features:** Age (51%), Annual Salary (48%), Gender (1%)

## Tech Stack
- **Backend:** Python + Flask
- **ML:** scikit-learn
- **Frontend:** Custom HTML/CSS/JS (dark theme)

## How to Use
1. Select your **gender**
2. Drag the **age** slider
3. Enter your **annual salary**
4. Click **Predict Now** — get an instant recommendation with probability breakdown!
