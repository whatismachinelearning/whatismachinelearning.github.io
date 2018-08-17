---
layout: post
title: Tổng quan về Machine Learning và các thách thức (Phần 1)
subtitle: Machine Learning là cái quái gì nhỉ?
categories: [Tổng quan, basic]
tags: [machine learning]
---

## Machine Learning là cái quái gì thế nhỉ?

Machine Learning (ML) đã xuất hiện cách đây hàng thập kỉ từ lúc có nhận diện kí tự quang học (Optical Character Recognition - OCR) và nhiều ứng dụng khác. Nhưng mới đây TV và báo đài mới nói nhiều về nó và thằng con của nó là Deep Learning nên người ta tưởng đây là lĩnh vực mới  chứ thực ra nó chỉ đang bùng nổ thôi.

## Một số định nghĩa cho Machine Learning

Machine Learning (Học Máy) là ngành khoa học (và nghệ thuật) về lập trình cho máy tính để chúng có thể học từ dữ liệu.
Một số định nghĩa khác:
> “Machine Learning là một phạm trù học thuật trong đó cung cấp cho máy tính khả năng học mà không cần phải được lập trình rõ ràng” – **Arthur Samuel, 1959.**

Định nghĩa kiểu kĩ thuật:

> “Một chương trình máy tính được gọi là học từ kinh nghiệm E với với nhiệm vụ T và một vài thước đo hiệu suất P, nếu hiệu suất của nó trên T khi được đo bởi P được cải thiện dần theo E.” – **Tom Mitchell, 1997.**

Mình xin trích một ví dụ trong sách **"Hands-on Machine Learning with sklearn and Keras"**:

>Ví dụ bạn có một chương trình ML giúp lọc thư rác từ việc học cách đánh dấu các thư rác mẫu và các thư bình thường mẫu. Tập hợp các "mẫu" mà hệ thóng sử dụng để học được gọi là *tập đào tạo* (training set) và mỗi một mẫu được gọi là *điểm dữ liệu* (training instace hoặc sample).
>
> Ở đây T là nhiệm vụ đánh dấu spam cho email mới, kinh nghiệm E là dữ liệu đào tạo, và thước đo P cần được thiết lập trước, ví dụ như tỉ lệ các email được đánh dấu đúng (cái này gọi là độ chính xác - *accuracy*).

## Machine Learning vs Hard code

*Hard code* có thể hiểu là việc viết code một cách hoàn toàn thủ công. Về cơ bản Machine learning ra đời để thay thế cho hàng loạt câu lệnh điều kiện (rules) được hard coded, khi dữ liệu thay đổi các điều kiện không còn thể hiện được tốt chức năng của mình nữa.

Machine learning cũng được dùng với các bài toán chưa có lời giải tối ưu bằng các phương pháp lập trình truyền thống. Ví dụ với bài toán đơn giản là nhận diện chữ "One" và "Two", người ta nhận thấy chữ "Two" khi đọc có cao độ cao hơn, nên việc hard code phân biệt chỉ hai âm thanh trên khá dễ dàng. Thế nhưng với hàng trăm nghìn từ thì việc hard code gần như bất khả thi.