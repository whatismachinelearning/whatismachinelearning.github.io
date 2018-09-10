---
title: So sánh Batch Learning và Online Learning
date: 2018-08-22 00:00:00 +07:00
categories:
- Tổng quan
tags:
- machine learning
- basic
layout: post
subtitle: Tại sao chúng lại được đem ra so sánh với nhau
author: yennhi
series: Tổng quan về Machine Learning và các thách thức
share-img: /media/tong-quan/batman-kmeans.png
---

Khi mình đọc đến phần phân loại các hệ thống Machine Learning (ML) mình cũng đã đưa ra câu hỏi này. Lẽ nào hai em nó có duyên nghiệp từ kiếp trước vì khi đọc tên (Học hàng loạt và học trực tuyến) thấy hai em nó chẳng có gì liên quan đến nhau cả. Bài viết này có lẽ sẽ giúp các bạn giải đáp thắc mắc này.

Bắt đầu đi vào cụ thể hai em nó là cái gì nào!

## BATCH LEARNING

![Batch learning](/media/tong-quan/training-model.png)
*Training - Nguồn: <https://www.slideshare.net/queirozfcom/online-machine-learning-introduction-and-examples>*

Trước hết, cứ theo bảng chữ cái mà táng, nói về bạn Batch Learning (có thể dịch là Học hàng loạt). Đại loại là mọi người đưa toàn bộ dữ liệu training vào hệ thống. Nghe cũng thấy là ăn tham thì nuốt không trôi rồi. Việc học kiểu này khá tốn thời gian và tài nguyên của máy tính thậm chí có khả năng vượt quá bộ nhớ của máy tính. Mà thời gian và tài nguyên đều rất tốn tiền. 

Sau khi đã được training rồi, hệ thống này sẽ chạy trên các sản phẩm mà không học thêm nữa, nói cách khác là nó học được gì rồi thì dùng cái đó, có cho thêm dữ liệu nào cũng chẳng học, có nghĩa cũng không ảnh hưởng đến đầu ra luôn. Vì thế cũng có thể nói hệ thống thường thực hiện ngoại tuyến (khi không có dữ liệu động truyền vào hệ thống) hay dịch ra tiếng Anh cho sang chảnh là Offline Learning (giờ thấy hơi liên quan đến nhau rồi đấy!!!).

Vậy nếu muốn hệ thống Batch Learning này học những dữ liệu mới thì phải làm sao đây? Thì lại làm lại từ đầu chứ sao. Huấn luyện lại trên tập dữ liệu từ đầu (tập này giờ có cả dữ liệu mới và dữ liệu cũ nhé), tạo ra phiên bản  mới của hệ thống, sau đó dừng chạy cái cũ, thay cái hệ thống mới này vào chạy tiếp. Nghe có vẻ lại tốn thời gian. May mắn rằng, toàn bộ quá trình huấn luyện, đáng giá và khởi chạy một hệ thống ML có thể dễ dàng thực hiện một cách tự động. Vậy nên, cứ cập nhật dữ liệu, huấn luyện một version thứ n của hệ thống, rồi chạy hệ thống này khi cần thiết thôi.

![Batch learning](/media/tong-quan/offline-learning.png)
*Offline Learning*

## ONLINE LEARNING

Giờ thì đến thanh niên còn lại Online Learning hay có thể gọi là Học trực tuyến. Khác với bạn trên ăn tham, bạn này ăn rất từ từ từng miếng một, có thể là instance (điểm dữ liệu), hay minibatch (nhóm nhỏ). 

Chính vì thế, hệ thống kiểu này thường có thể học ngay dữ liệu mới. Mỗi bước học vì đã được chia nhỏ đầu vào nên cũng không tốn kém như bạn trên. Hơn nữa, vì nó cập nhật dữ liệu liên tục, nên nếu bạn cũng có nhu cầu “replay” dữ liệu, thì có thể xóa dữ liệu đã học xong rồi đi. Rất chi là tiết kiệm không gian nhớ.

Trong hệ thống kiểu này lại có một tham số là learning rate (tham số này bạn có thể gặp ở các các bài toán sử dụng Gradient Descent), cái này đo tốc độ thích ứng của hệ thống với việc thay đổi của dữ liệu, và tỷ lệ này do chính bạn đặt ra cho hệ thống. Nếu tỷ lệ này cao, hệ thống của bạn sẽ đáp ứng nhanh với dữ liệu mới, và đồng nghĩa với nó không thể khác là việc quên nhanh dữ liệu cũ. Ngược lại thì hệ thống sẽ học chậm hơn, nhớ lâu hơn chính vì thế cũng ít nhạy cảm với nhiễu hơn.

Nãy giờ toàn thấy cái hay của bạn này. Có một điều không hay đây. Khi dữ liệu xấu (lắm nhiễu chẳng hạn) được đưa vào hệ thống, hiệu suất của hệ thống sẽ giảm đáng kể. Vậy thì, khi làm việc với hệ thống kiểu này mà thấy hiệu suất giảm, bạn nên nhanh chóng dừng việc học lại và có thể trở về trạng thái trước đó nếu cần thiết. Bạn cũng có thể giám sát dữ liệu đầu vào bằng các thuật toán phát hiện bất thường để tránh dữ liệu xấu vào hệ thống.

## VẬY KHI NÀO THÌ DÙNG BATCH LEARNING, KHI NÀO THÌ DÙNG ONLINE LEARNING.

Nếu tập dữ liệu của bạn không quá lớn, bộ nhớ của bạn đủ chứa lượng dữ liệu kia, và đặc biệt hệ thống của bạn không cần thiết phải thích ứng nhanh với dữ liệu thay đổi thì bạn có thể xem xét việc dùng Batch Learning. Còn nếu dữ liệu của bạn lớn, hoặc bộ nhớ của bạn lại ít- chẳng hạn như con rover trên sao Hỏa- và hệ thống của bạn lại cần phải thích ứng nhanh với dữ liệu (như dự đoán giá cổ phiếu chẳng hạn), thì Online Learning nên là lựa chọn được ưu tiên. Đặc biệt, Online Learning cực hiệu quả khi phải huấn luyện hệ thống trên một tập dữ liệu quá lớn mà không thể vừa với bộ nhớ chính của bất kì máy nào, điều này còn được gọi là Out-of-core Learning- học tập ngoài lõi

## CHỐT!

Kể cả đối với Online learning, thì thường hệ thống của bạn sẽ học xong xuôi trên các dữ liệu rồi mới hoạt động, nghĩa là không học khi hệ thống đang chạy, do dó bạn lại thấy Online Learning có vẻ là cái tên không hợp lý lắm. Vì thế nên bạn thường được gọi theo một cái tên khác là incremental learning- học gia tăng. Nhưng, không phải vì cái tên, mà bởi vì các đặc điểm của Batch Learning và Online Learning thường đối lập nhau, nên hai bạn này  mới hay được so sánh và đi chung với nhau như thế nhé!