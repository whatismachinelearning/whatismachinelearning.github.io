---
layout: post
title: Tổng quan về Machine Learning và các thách thức (Phần 1)
subtitle: Machine Learning là cái quái gì thế nhỉ?
categories:
  - Tổng quan
  - basic
tags:
  - machine learning
published: true
---
Machine Learning (ML) đã xuất hiện cách đây hàng thập kỉ từ lúc có nhận diện kí tự quang học (Optical Character Recognition - OCR) và nhiều ứng dụng khác. Nhưng mới đây TV và báo đài mới nói nhiều về nó và thằng em của nó là Deep Learning nên người ta tưởng đây là lĩnh vực mới  chứ thực ra nó chỉ đang bùng nổ thôi.

## AI và "The hype"

AI - *Artificial Intelligence* hay *Trí tuệ nhân tạo* đang là cụm từ được nhắc tới rất nhiều trên mạng xã hội cũng như các phương tiện thông tin đại chúng. Rõ ràng có rất nhiều thứ mà "AI" đã và đang làm được khiến con người phải kinh ngạc và sức mạnh của nó là khỏi bàn cãi. Tuy nhiên trên thực tế trên thế giới chưa có AI theo đúng cái nghĩa người ta đang kì vọng - một cỗ máy có trí thông minh thực thụ.

Để nói rõ hơn thì AI được chia thành 2 loại:
  - Strong AI hay True AI: Một cỗ máy *"thông minh"* có kĩ năng, sự thông minh và khéo léo **ít nhất** là như con người.[^1]
  - Weak AI hay Narrow AI: Một cỗ máy *"thông minh"* có khả năng giải quyết một công việc cụ thể .[^2]

Thực sự mình cũng không muốn viết ra từ "thông minh" vì mấu chốt của AI chính là ở việc định nghĩa từ "thông minh" này. Thế nào là thông minh, và làm thế nào để dạy được máy điều đó, có lẽ chúng ta vẫn còn rất nhiều điều phải làm trước khi có thể làm ra được một strong AI thực thụ. Phần lớn (nếu không muốn nói là tất cả) những ứng dụng mà chúng ta được biết là sản phẩm của một Weak AI (Siri hay Google Assistant chính là các ví dụ điển hình). 

Trong Blog này chúng mình sẽ giới hạn ở Machine Learning và Deep Learning thôi vì kiến thức cũng có hạn. Nhưng trước khi nhảy vào deep learning thì phải ghé qua Machine Learning cái đã nhỉ. Chẳng ai không biết bơi mà lại biết lặn (xong phải nổi lên được nha) đúng không :D

## Một số định nghĩa cho Machine Learning

{% include image.html url="/img/tong-quan/machine-learning-image.png" description="Nguồn: http://www.zarantech.com/blog/an-introduction-to-machine-learning-why-it-matters/" %}

<!-- ![Nguồn: http://www.zarantech.com/blog/an-introduction-to-machine-learning-why-it-matters/](/img/tong-quan/machine-learning-image.png) -->

Machine Learning (Học Máy) là ngành khoa học (và nghệ thuật) về lập trình cho máy tính để chúng có thể học từ dữ liệu.
Một số định nghĩa khác:
> “Machine Learning là một phạm trù học thuật trong đó cung cấp cho máy tính khả năng học mà không cần phải được lập trình rõ ràng” – **Arthur Samuel, 1959.**

Định nghĩa kiểu kĩ thuật:

> “Một chương trình máy tính được gọi là học từ kinh nghiệm E với với nhiệm vụ T và một vài thước đo hiệu suất P, nếu hiệu suất của nó trên T khi được đo bởi P được cải thiện dần theo E.” – **Tom Mitchell, 1997.**

Mình xin trích một ví dụ trong sách **“Hands-On Machine Learning with Scikit-Learn and TensorFlow[^3]”**:

Ví dụ bạn có một chương trình ML giúp lọc thư rác từ việc học cách đánh dấu các thư rác mẫu và các thư bình thường mẫu. Tập hợp các "mẫu" mà hệ thóng sử dụng để học được gọi là *tập đào tạo* (training set) và mỗi một mẫu được gọi là *điểm dữ liệu* (training instace hoặc sample).

Ở đây T là nhiệm vụ đánh dấu spam cho email mới, kinh nghiệm E là dữ liệu đào tạo, và thước đo P cần được thiết lập trước, ví dụ như tỉ lệ các email được đánh dấu đúng (cái này gọi là độ chính xác - *accuracy*).

## Machine Learning vs Hard code

*Hard code* có thể hiểu là việc viết code một cách hoàn toàn thủ công. Về cơ bản Machine learning ra đời để thay thế cho hàng loạt câu lệnh điều kiện (rules) được hard coded, khi dữ liệu thay đổi các điều kiện không còn thể hiện được tốt chức năng của mình nữa.

Vẫn lấy ví dụ về bài toán phân loại mail rác. Là một nhà cung cấp dịch vụ email, bạn nhận thấy rằng gần đây người dùng của bạn có xu hướng đánh dấu các email có chứ cụm từ "4U", "Miễn phí", "trúng thưởng"... là mail rác. Bạn tạo bộ lọc chứa các cụm từ trên, thế nhưng bọn "mail tặc" cũng không vừa, chúng nắm được quy luật chặn mail của bạn và thay đổi các cụm từ trên thành "Cho bạn", "free", "trúng lớn"... và thế là công cốc, bạn lại phải thêm một loạt điều kiện mới. Cứ thế thì cuộc chiến sẽ là không hồi kết.

Nhưng thay vào đó, ML tự động nhận thấy sự ra tăng bất thường của các email chứa các cụm từ "nóng" được người dùng đánh dấu và tự động thêm vào bộ lọc của chúng mà không cần bạn ra tay. Khỏe re.

Machine learning cũng được dùng với các bài toán chưa có lời giải tối ưu bằng các phương pháp lập trình truyền thống. Ví dụ với bài toán đơn giản là nhận diện âm "One" và "Two", người ta nhận thấy chữ "Two" khi đọc có cao độ cao hơn, nên việc hard code phân biệt chỉ hai âm thanh trên khá dễ dàng. Thế nhưng với hàng trăm nghìn từ thì việc phân loại chúng bằng hard code gần như bất khả thi.

## Tóm lại thì...
Machine Learning được dùng cho:
  - Các vấn đề đã có lời giải nhưng phức tạp và cần quá nhiều điều kiện
  - Các bài toán phức tạp không có lời giải đủ tốt bằng phương pháp lập trình truyền thống
  - Môi trường dữ liệu thay đổi liên tục
  - Cần cái nhìn tổng quan về các vấn đề phức tạp và lượng dữ liệu lớn.

## Các ứng dụng điển hình
Machine Learning có vô vàn ứng dụng, đặc biệt là ở kỉ nguyên bùng nổ dữ liệu như hiện nay. Tuy nhiên sau khi tham khảo mình cũng xin mạn phép đưa ra vài ứng dụng điển hình của Machine Learning như sau:
- Marketing và Sale
- Chăm sóc sức khỏe
- Ngân hàng và tài chính
- Giao thông
- Giáo dục
- Máy tìm kiếm

Vài gạch đầu dòng là chưa đủ, với sức mạnh của mình Machine Learning đang dần dấn thân vào tất cả các lĩnh vực truyền thống và mang lại sức mạnh mới cho chúng.

### Happy suffing!

[^1]: Hands-On Machine Learning with Scikit-Learn and TensorFlow by Aurélien Géron (O’Reilly). Copyright 2017 Aurélien Géron, 978-1-491-96229-9.
[^2]: https://en.wikipedia.org/wiki/Strong_AI
[^3]: https://en.wikipedia.org/wiki/Weak_AI