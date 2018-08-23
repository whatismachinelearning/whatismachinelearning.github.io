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
series: tong-quan
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

Một cách phân loại rất phổ biến khác là phân loại theo nhóm chức năng. Sau đây là list các nhóm thuật toán nên học (tham khảo từ vài nguồn). Sau này mình sẽ cố gắng viết về tất cả các thuật toán dưới đây:

### Bài toán Regression (hồi quy)

Về định nghĩa:

>Phân tích hồi quy là một phân tích thống kê để xác định xem các biến độc lập (biến thuyết minh) quy định các biến phụ thuộc (biến được thuyết minh) như thế nào. - [Wikipedia](https://vi.wikipedia.org/wiki/Ph%C3%A2n_t%C3%ADch_h%E1%BB%93i_quy)

Nói cách khác các bài toán hồi quy quan tâm đến sự tương quan giữa các biến (features) và điều chỉnh dần theo dữ liệu đầu vào.

Một số thuật toán Regression nổi tiếng:
- Ordinary Least Squares Regression (OLSR)
- Linear Regression
- Logistic Regression
- Stepwise Regression

### Các thuật toán Instance based (học theo điểm dữ liệu)

Nói ngắn gọn thì học theo điểm dữ liệu là học thuộc lòng dữ liệu input, rồi xác định output của điểm dữ liệu mới dựa trên **sự tương tự** của nó với các điểm cũ. Các bài toán kiểu này thường có thời gian training ngắn hơn thời gian dự đoán.

Một vài thuật toán nổi tiếng của loại này:
- k-Nearest Neighbor (kNN)
- Learning Vector Quantization (LVQ)

### Các bài toán Classification (Phân loại)
- Linear Classifier
- Support Vector Machine (SVM)
- Kernel SVM
- Sparse Representation-based classification (SRC)

### Các thuật toán Regularize (không biết phải dịch như nào cho hay)

- Ridge Regression
- Least Absolute Shrinkage and Selection Operator (LASSO)
- Least-Angle Regression (LARS)

### Các thuật toán Bayesian

- Naive Bayes
- Gaussian Naive Bayes

### Các thuật toán Clustering (Phân cụm)

- k-Means clustering
- k-Medians
- Expectation Maximization (EM)

### Artificial Neural Network Algorithms
- Perceptron
- Softmax Regression
- Multi-layer Perceptron
- Back-Propagation

### Các bài toán Tree based (Cây quyết định)

- Classification and Regression Tree (CART)
- Iterative Dichotomiser 3 (ID3)
- C4.5 and C5.0
- Chi-squared Automatic Interaction Detection (CHAID)
- Decision Stump
- M5
- Conditional Decision Trees

Và còn RẤT NHIỀU các thuật toán khác, mình xin được phép giới thiệu dần trong các bài viết sau.

## Lời cuối

Trong phần này chúng ta đã có một cái nhìn tổng quan về thế giới thuật toán Machine Learning. Phần cuối của serie này mình sẽ nói về các thách thức và các vấn đề gặp phải mà người ta hay gặp phải trong thực tế. Bài viết sau mình xin phép được dịch lại từ nhiều nguồn vì chưa đủ kinh nghiệm để chia sẻ với các bạn.