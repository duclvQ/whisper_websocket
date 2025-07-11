from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


model_name = "VietAI/envit5-translation"
tokenizer = AutoTokenizer.from_pretrained(model_name)  
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.cuda()

inputs = [
    """vi: Liên quan tới công tác chuẩn bị đại hội đảng các cấp theo chỉ thị số 45, Chủ tịch nước yêu cầu Ban thường vụ 2 tỉnh cần tiếp tục phối hợp chặt chẽ, chuẩn bị thật tốt các nội dung văn kiện, trong đó cần chú ý quán triệt bổ sung những định hướng, những điểm mới của Trung ương trong dự thảo báo cáo chính trị, trong công tác xây dựng đảng, trong chiến lược phát triển kinh tế xã hội 10 năm 2021-2030, phương hướng phát triển kinh tế xã hội 5 năm 2026-2030, đồng thời văn kiện đại hội cũng phải thể hiện là sự kết tinh trí tuệ.
    """
    ]

outputs = model.generate(tokenizer(inputs, return_tensors="pt", padding=True).input_ids.to('cuda'), max_length=512)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

# ['en: VietAI is a non-profit organization with the mission of nurturing artificial intelligence talents and building an international - class community of artificial intelligence experts in Vietnam.',
#  'en: According to the latest LinkedIn report on the 2020 list of attractive and promising jobs, AI - related job titles such as AI Specialist, ML Engineer and ML Engineer all rank high.',
#  'vi: Nhóm chúng tôi khao khát tạo ra những khám phá có ảnh hưởng đến mọi người, và cốt lõi trong cách tiếp cận của chúng tôi là chia sẻ nghiên cứu và công cụ để thúc đẩy sự tiến bộ trong lĩnh vực này.',
#  'vi: Chúng ta đang trên hành trình tiến bộ và dân chủ hoá trí tuệ nhân tạo thông qua mã nguồn mở và khoa học mở.']
