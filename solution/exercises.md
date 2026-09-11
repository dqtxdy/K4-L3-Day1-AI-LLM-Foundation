# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

## Block 1 — API Cơ Bản

### Câu 1.1 — Độ nhạy của temperature

Trong lần chạy này, temperature 0.0, 0.5 và 1.0 đều trả lời về hang Sơn Đoòng,
với cách diễn đạt khá giống nhau; ở 1.5, model chuyển sang một sự thật về cà
phê Việt Nam. Như vậy temperature cao làm lựa chọn nội dung đa dạng hơn, nhưng
độ trễ 3.782s, 3.089s, 2.619s và 1.777s không cho thấy quy luật đáng tin cậy
vì còn phụ thuộc vào mạng và tải API.

### Câu 1.2 — Chọn temperature cho sản phẩm

Em sẽ bắt đầu với temperature khoảng 0.2–0.3. Chatbot hỗ trợ khách hàng cần
trả lời nhất quán, chính xác và ít bịa thêm; mức thấp vẫn đủ linh hoạt để diễn
đạt tự nhiên. Nếu sản phẩm có nhiều câu hỏi sáng tạo hơn, em sẽ thử nghiệm
trên tập đánh giá trước khi tăng giá trị này.

### Câu 1.3 — Đánh đổi chi phí

Với phần output của workload này, GPT-4o có giá 0.010 USD/1K token còn
GPT-4o-mini là 0.0006 USD/1K token, nên GPT-4o đắt khoảng 16.7 lần. 10.000
người dùng × 3 lần × 350 token là khoảng 10,5 triệu token output/ngày, tương
ứng khoảng 105 USD/ngày với GPT-4o và 6,3 USD/ngày với mini (chưa tính input).
GPT-4o đáng dùng cho phân tích phức tạp hoặc câu trả lời có rủi ro cao; mini
phù hợp với FAQ, phân loại yêu cầu và các tác vụ lặp lại.

## Block 2 — System Prompt & Token

### Câu 2.1 — Sức mạnh của persona

Trong lần chạy này, prompt dành cho trẻ 8 tuổi dùng ví dụ một cuốn sổ chung để
ghi việc mượn đồ chơi, với từ vựng và câu giải thích đơn giản. Prompt dành cho
chuyên gia tài chính tạo câu trả lời dài hơn, dùng các thuật ngữ như cấu trúc
dữ liệu phân tán, phi tập trung, hash và sự đồng thuận. System prompt đã thay
đổi rõ đối tượng, mức chi tiết và cách chọn ví dụ dù câu hỏi người dùng không đổi.

### Câu 2.2 — tiktoken vs đếm từ

Với đoạn tiếng Việt 107 từ em dùng, `count_tokens` theo encoding của gpt-4o
cho 129 token, còn công thức `107 / 0.75` cho 142.67 token; công thức thô cao
hơn thực tế 9.58%. Token là các mảnh subword chứ không phải từ; dấu tiếng Việt,
cách ghép âm tiết và độ phổ biến của từ ảnh hưởng đến cách tách token, nên số
token có thể khác tiếng Anh cùng độ dài. Vì vậy `số từ / 0.75` chỉ nên dùng để
ước lượng nhanh, không thay cho bộ mã hóa thật.

## Block 3 — Streaming & Độ Bền

### Câu 3.1 — Trải nghiệm người dùng với streaming

Streaming quan trọng khi phản hồi có thể dài hoặc model cần vài giây để sinh
xong, chẳng hạn chatbot, trợ lý viết và giao diện tương tác; người dùng thấy
ứng dụng đã hoạt động ngay thay vì phải chờ toàn bộ câu trả lời. Non-streaming
phù hợp khi cần nhận một kết quả hoàn chỉnh để xử lý tiếp, lưu vào cơ sở dữ
liệu, kiểm tra toàn bộ nội dung trước khi hiển thị, hoặc khi phản hồi rất ngắn
và việc quản lý stream không đem lại lợi ích đáng kể.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?

Exponential backoff làm các client thất bại tạm thời giãn dần thời điểm retry,
ví dụ 0.1, 0.2, 0.4 giây, nên server có thời gian phục hồi và hệ thống giảm
traffic dồn thêm khi đang quá tải. Delay cố định khiến hàng nghìn client retry
đúng cùng một thời điểm, tạo hiện tượng thundering herd: đợt retry mới lại
làm server nghẽn và có thể gây lỗi dây chuyền. Thực tế nên kết hợp thêm jitter
ngẫu nhiên để các thời điểm retry không trùng nhau.

## Block 4 — Mini-Project

### Câu 4.1 — Thiết kế persona

Persona em chọn là: “Bạn là trợ giảng thân thiện của khóa AI. Hãy giải thích
rõ ràng, ngắn gọn bằng tiếng Việt, ưu tiên ví dụ thực tế và nói thẳng khi bạn
không chắc chắn.” Cụm “ngắn gọn bằng tiếng Việt” giúp câu trả lời phù hợp với
người học và không lan man; “ưu tiên ví dụ thực tế” biến khái niệm trừu tượng
thành thứ dễ áp dụng. Yêu cầu nói rõ khi không chắc chắn giúp giảm cảm giác
chắc chắn giả khi model thiếu thông tin.

### Câu 4.2 — Hạn chế & cải thiện

Hạn chế lớn nhất là history chỉ giữ ba lượt, nên trợ lý có thể quên một quyết
định hoặc định nghĩa đã nói ở lượt thứ tư trở về trước. Em sẽ cải thiện bằng
cách tóm tắt các lượt cũ: trước khi cắt history, gửi phần hội thoại cũ cho một
hàm tóm tắt, lưu bản tóm tắt cùng các thông tin quan trọng, rồi đưa bản tóm tắt
vào system/context ở những lượt sau. Cách này giữ được ngữ cảnh dài hơn mà
không làm số token input tăng tuyến tính theo toàn bộ lịch sử.
