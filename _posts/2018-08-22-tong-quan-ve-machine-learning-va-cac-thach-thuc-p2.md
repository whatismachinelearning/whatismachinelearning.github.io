---
title: Tổng quan về Machine Learning và các thách thức (Phần 2)
date: 2018-08-22 00:00:00 +07:00
categories:
- Tổng quan
tags:
- machine learning
- basic
layout: post
subtitle: Cách người ta phân loại các thuật toán Machine Learning
author: levulinh
series: Tổng quan về Machine Learning và các thách thức
share-img: /media/tong-quan/batman-kmeans.png
---

Có rất nhiều thuật toán Machine Learning, nhiều đến mức chỉ nghe người khác nói chuyện về tên chúng thôi là đủ bạn đau đầu rồi (giống mình). Tuy vậy cũng giống như học tiếng Anh, chỉ có 20% số từ được dùng trong 80% các câu nói hàng ngày của bạn. Các thuật toán Machine Learning có rất nhiều điểm tương đồng được nhóm vào cùng nhóm, điều này có thể giúp bạn trả lời được câu hỏi muôn thuở: "Mình muốn học Machine Learning thì bắt đầu từ đâu nhỉ?".

Trong bài viết này mình sẽ giới thiệu hai cách mà người ta thường dùng để phân loại các bài toán Machine Learning: Theo cách học và theo tương đồng.

Hi vọng sau bài viết này các bạn sẽ tự lập cho mình một check list các thuật toán Machine Learning cơ bản nhất để bắt đầu. Bắt đầu tour thôi.

## Phân loại theo cách học

Có nhiều cách mà các thuật toán ML tương tác với môi trường dữ liệu (có tài liệu gọi là **Experience**). Trong đó người ta thường chia các thuật toán Machine Learning thành 4 loại: Supervised, Unsupervised, Semi-supervised và Reinforcement Learning.

### Supervised Learning

![Linear Regression](/media/tong-quan/lin-reg.png)
*Ví dụ về Linear Regression - Nguồn: Wikipeia*

Supervised Learning, hay học có giám sát, là loại bài toán phổ biến nhất trong các thuật toán Machine Learning. Dữ liệu của bài toán này là các cặp (data, label) trong đó label là *output* tương ứng với mỗi data. Nhớ lại ví dụ về [Bài toán phân loại mail rác](/2018-08-17-tong-quan-ve-machine-learning-va-cac-thach-thuc-p1/#machine-learning-vs-hard-code) đã nhắc tới ở bài trước, nội dung mail là data và cờ do người dùng gắn là label.

Mô hình được bài toán xây dựng lên được yêu cầu liên tục dự đoán và điều chỉnh tới khi đạt sai số nhỏ (hoặc độ chính xác lớn) chấp nhận được.

Một số bài toán phổ biến của loại này có thể kể đến Regression (hồi quy) và Classification (phân loại).

List một số thuật toán nổi tiếng:
- Linear Regression
- Logistic Regression
- Support Vector Machine
- PLA Neural Network

### Unsupervised learning

Học không giám sát ít phổ biến hơn học có giám sát, nhưng không phải vì thế mà nó không quan trọng. Dữ liệu không được gắn nhãn, chỉ có các thuộc tính (đôi khi gọi là đặc tính), thuật toán thực hình các tính toán dựa trên các thuộc tính của dữ liệu để tìm ra đặc điểm chung. Các bài toán của loại này gồm có: Clustering (phân cụm), Visualization (hình dung hóa), dimentionality reduction (giảm chiều dữ liệu), anomaly detection (phát hiện bất thường) và association rule learning (học quy luật liên kết dữ liệu).

Về ứng dụng có thể kể tới nhóm các thuật toán gợi ý nội dung như trên Youtube, Facebook hay Lazada...

Thuật toán cơ bản nhất của loại này có thể nói đến K-means clustering.

![K-means](/media/tong-quan/batman-kmeans.png)
*Ứng dụng của K-means vào tìm màu chính trong ảnh - Nguồn: <https://www.pyimagesearch.com>*

### Semi-supervised learing

Semi-supervised learing hay còn gọi là Học bán giám sát. Chỉ một lượng nhỏ trong số dữ liệu của loại này là có nhãn, vì vậy nó là lai của hai loại trên. Lấy ví dụ như thuật toán tự động tag khuôn mặt của facebook: Rất nhiều ảnh và rất nhiều khuôn mặt của người dùng được thu thập về, nhưng chỉ số ít được người dùng gắn nhãn (coi như là nguồn đáng tin cậy), số còn lại facebook chỉ có thể dự đoán rằng một người này xuất hiện trong nhiều ảnh nhưng không rõ là ai. Như vậy muốn gắn thẻ một cách tự động tất cả các khuôn mặt thì phải có sự tham gia của Học giám sát để gắn nhãn khuôn mặt và Học không giám sát để nhóm các khuôn mặt giống nhau với nhau.

### Reinforcement learning

Học tăng cường -  dữ liệu của loại này chính là ngữ cảnh, quyết định và sự đánh giá quyết định đó. Nói chung rất là hay cơ mà cũng rất khó vì trích xuất được dữ liệu của loại này không dễ.

Ví dụ về học tăng cường có thể xem qua video thú vị sau:

{% include video.html id="3bhP7zulFfY" %}

## Phân loại theo chức năng

{: .box-note}
**Chỉnh sửa:** Sau khi được chỉ ra có vấn đề trong bài viết mình đã quyết định sửa lại từ "phân loại theo chức năng" thành "phần loại theo sự tương tự". Phần phía dưới sẽ được trích trực tiếp từ blog [Machinelearningmastery.com](https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/) (Mình đã đọc kĩ Terms of service của họ rồi nhé :p)

Một cách phân loại rất phổ biến khác là phân loại theo sự tương tự về chức năng. Sau đây là list các nhóm thuật toán nên học mà mình lượm được ở nguồn bên trên. Sau này mình sẽ cập nhật link tới các bài viết dần (nếu có), còn bây giờ mọi người xem tạm cái list này nhé.

### Regression Algorithms
- Ordinary Least Squares Regression (OLSR)
- Linear Regression
- Logistic Regression
- Stepwise Regression
- Multivariate Adaptive Regression Splines (MARS)
- Locally Estimated Scatterplot Smoothing (LOESS)

### Instance-based Algorithms
- k-Nearest Neighbor (kNN)
- Learning Vector Quantization (LVQ)
- Self-Organizing Map (SOM)
- Locally Weighted Learning (LWL)

### Regularization Algorithms
- Ridge Regression
- Least Absolute Shrinkage and Selection Operator (LASSO)
- Elastic Net
- Least-Angle Regression (LARS)

### Decision Tree Algorithms
- Classification and Regression Tree (CART)
- Iterative Dichotomiser 3 (ID3)
- C4.5 and C5.0 (different versions of a powerful approach)
- Chi-squared Automatic Interaction Detection (CHAID)
- Decision Stump
- M5
- Conditional Decision Trees

### Bayesian Algorithms
- Naive Bayes
- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Averaged One-Dependence Estimators (AODE)
- Bayesian Belief Network (BBN)
- Bayesian Network (BN)

### Clustering Algorithms
- k-Means
- k-Medians
- Expectation Maximisation (EM)
- Hierarchical Clustering

### Association Rule Learning Algorithms
- Apriori algorithm
- Eclat algorithm

### Artificial Neural Network Algorithms
- Perceptron
- Back-Propagation
- Hopfield Network
- Radial Basis Function Network (RBFN)

### Deep Learning Algorithms
- Deep Boltzmann Machine (DBM)
- Deep Belief Networks (DBN)
- Convolutional Neural Network (CNN)
- Stacked Auto-Encoders

### Dimensionality Reduction Algorithms
- Principal Component Analysis (PCA)
- Principal Component Regression (PCR)
- Partial Least Squares Regression (PLSR)
- Sammon Mapping
- Multidimensional Scaling (MDS)
- Projection Pursuit
- Linear Discriminant Analysis (LDA)
- Mixture Discriminant Analysis (MDA)
- Quadratic Discriminant Analysis (QDA)
- Flexible Discriminant Analysis (FDA)

### Ensemble Algorithms
- Boosting
- Bootstrapped Aggregation (Bagging)
- AdaBoost
- Stacked Generalization (blending)
- Gradient Boosting Machines (GBM)
- Gradient Boosted Regression Trees (GBRT)
- Random Forest

Vẫn còn nhưng nhiêu đây đủ chóng mặt rồi nhỉ :)) hi vọng các bạn sẽ chọn được một thuật toán của mỗi nhóm để bắt đầu học nhé.

{: .box-warning}
*Copyright issues:* In case you think that the contents on this post violate your copyright terms, please contact me and I will reply you ASAP.

## Lời cuối

Trong phần này chúng ta đã có một cái nhìn tổng quan về thế giới thuật toán Machine Learning. Phần cuối của serie này mình sẽ nói về các thách thức và các vấn đề gặp phải mà người ta hay gặp phải trong thực tế. Bài viết sau mình xin phép được dịch lại từ nhiều nguồn vì chưa đủ kinh nghiệm để chia sẻ với các bạn.