# 📈 Smart Strategies for an Effective Marketing Campaign Challenge 🚀

## 📸 Context
This project was created as part of a job application challenge to showcase my skills. It utilizes a dataset that cannot be shared due to (potential) confidentiality reasons, and no PRs are allowed.

## 📁 How to execute 

 1. Insert 4 valid .csv datasets in the test/train folders.

    1.1 Training Session:
        ```
        user_id,session_id,timestamp,device_type,browser,operating_system,ip_address,country,search_query,page_views,session_duration
        ```

     1.2 Training User:
        ```
          user_id;age;abandoned_cart;user_category;test_id
        ```

     1.3 Testing: 
        ```
          Almost all the same, replace marketing_target for test_id
        ```

 2. Run: train, validate and predict.
    ```
    python model.py
    ```

## ✅ Tasks

"**Calculate the `marketing_target` Column**: Utilize the provided datasets to compute and assign a `marketing_target` value for each user. This column should reflect the predicted engagement level of the user with the marketing campaign, with values assigned as follows:
  - `1` for Low engagement
  - `2` for Medium engagement
  - `3` for High engagement"