
# Food Recommendation System

A Restaurant and Dish Recommender system using user profile, order history and ratings based similarity computation.

The users are presented with a choice of three recommendation models. The first one is a content-based filtering model, whereas the second and third are keyword-based filtering models. 
In the third model, recommendations are modified based on user feedback.








## Tech Stack

**Frontend:** Javascript, HTML, CSS 

**Backend:** Python , Django , MySQL / Postgres SQL

**others** scikit-learn , numpy , pandas 


## Overview : 

- Utilizing person preference to recommend more personalized food items.

- Content based recommendation system recommends items based on the content of items. I.e.   features of items
- First model is based on using TF-IDF vectorization and cosine similarity.
- Second model is based on the keyword extracted from dishes like ingredients, flavor, profile etc.
- Third model is based on penalization of keywords and promotion and demotion of certain keywords by feature vectors of the user based on the feedback received from users.

