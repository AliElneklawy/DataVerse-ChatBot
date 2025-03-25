from manim import *

class DataVerseChatBot(Scene):
    def construct(self):
        # Theme colors
        primary_color = "#3498db"  # Blue
        secondary_color = "#2ecc71"  # Green
        accent_color = "#e74c3c"  # Red
        neutral_color = "#95a5a6"  # Gray
        highlight_color = "#f39c12"  # Orange
        
        # Title and Introduction
        self.intro_section(primary_color)
        
        # Data Extraction and Processing
        self.data_extraction_section(primary_color, secondary_color)
        
        # LLM Integration and RAG
        self.llm_integration_section(primary_color, highlight_color)
        
        # Monitoring and Uncertainty Detection
        self.monitoring_section(primary_color, accent_color)
        
        # Chat Interfaces
        self.chat_interfaces_section(primary_color, secondary_color)
        
        # Dataset Creation and Model Training
        self.dataset_training_section(primary_color, highlight_color)
        
        # Architecture Overview
        self.architecture_overview(primary_color, secondary_color, accent_color, highlight_color)
        
        # Conclusion
        self.conclusion(primary_color)
    
    def intro_section(self, primary_color):
        # Title
        title = Text("DataVerse ChatBot", font_size=72, color=primary_color).to_edge(UP)
        subtitle = Text("AI-Driven Chat Interactions with Any Data Source", 
                        font_size=36).next_to(title, DOWN)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(2)
        
        # Brief description
        description = Text(
            "A powerful Python-based application that enables real-time, AI-driven chat interactions",
            font_size=28, line_spacing=1.5
        ).next_to(subtitle, DOWN, buff=1)
        
        self.play(Write(description))
        self.wait(2)
        
        # Show main features as icons with labels
        features = VGroup()
        
        # Create feature icons
        web_icon = SVGMobject("web").scale(0.5)
        file_icon = SVGMobject("file").scale(0.5)
        bot_icon = SVGMobject("bot").scale(0.5)
        llm_icon = SVGMobject("llm").scale(0.5)
        
        # For the actual implementation, you would need to create or use existing SVG files
        # Here I'm simulating with circles
        web_icon = Circle(radius=0.5, color=primary_color, fill_opacity=0.8)
        file_icon = Circle(radius=0.5, color=primary_color, fill_opacity=0.8)
        bot_icon = Circle(radius=0.5, color=primary_color, fill_opacity=0.8)
        llm_icon = Circle(radius=0.5, color=primary_color, fill_opacity=0.8)
        
        # Add text below icons
        web_text = Text("Web Crawling", font_size=20).next_to(web_icon, DOWN)
        file_text = Text("Data Extraction", font_size=20).next_to(file_icon, DOWN)
        bot_text = Text("Chat Interfaces", font_size=20).next_to(bot_icon, DOWN)
        llm_text = Text("LLM Integration", font_size=20).next_to(llm_icon, DOWN)
        
        # Group icons with their texts
        web_group = VGroup(web_icon, web_text)
        file_group = VGroup(file_icon, file_text)
        bot_group = VGroup(bot_icon, bot_text)
        llm_group = VGroup(llm_icon, llm_text)
        
        # Position icons in a row
        features = VGroup(web_group, file_group, bot_group, llm_group).arrange(RIGHT, buff=1.5)
        features.next_to(description, DOWN, buff=1)
        
        # Play animation
        self.play(FadeIn(features))
        self.wait(2)
        
        # Clear screen for next section
        self.play(
            FadeOut(title),
            FadeOut(subtitle),
            FadeOut(description),
            FadeOut(features)
        )
    
    def data_extraction_section(self, primary_color, secondary_color):
        # Section title
        section_title = Text("Data Extraction and Processing", font_size=56, color=primary_color).to_edge(UP)
        self.play(Write(section_title))
        self.wait(1)
        
        # Create a visualization of web crawling
        web_rect = Rectangle(height=3, width=4, fill_color=secondary_color, fill_opacity=0.3, stroke_color=primary_color)
        web_label = Text("Web Sources", font_size=36).next_to(web_rect, UP)
        web_group = VGroup(web_rect, web_label).to_edge(LEFT).shift(UP)
        
        # Create visualization for file extraction
        file_rect = Rectangle(height=3, width=4, fill_color=secondary_color, fill_opacity=0.3, stroke_color=primary_color)
        file_label = Text("File Formats", font_size=36).next_to(file_rect, UP)
        file_formats = VGroup(
            Text("PDFs", font_size=24),
            Text("DOCx", font_size=24),
            Text("CSV", font_size=24),
            Text("XLSX", font_size=24)
        ).arrange(DOWN, buff=0.3).move_to(file_rect)
        file_group = VGroup(file_rect, file_label, file_formats).to_edge(RIGHT).shift(UP)
        
        # Show web and file sources
        self.play(
            FadeIn(web_group),
            FadeIn(file_group)
        )
        self.wait(2)
        
        # Arrows from sources to processing
        web_arrow = Arrow(web_rect.get_bottom(), DOWN * 0.5, color=primary_color)
        file_arrow = Arrow(file_rect.get_bottom(), DOWN * 0.5, color=primary_color)
        
        # Processing box in the middle
        process_rect = Rectangle(height=2, width=6, fill_color=primary_color, fill_opacity=0.5, stroke_color=primary_color)
        process_rect.next_to(VGroup(web_arrow, file_arrow), DOWN)
        process_label = Text("Data Processing", font_size=36).next_to(process_rect, UP)
        
        process_group = VGroup(process_rect, process_label)
        
        # Animate arrows and processing
        self.play(
            Create(web_arrow),
            Create(file_arrow)
        )
        self.play(FadeIn(process_group))
        
        # Storage box below
        storage_rect = Rectangle(height=2, width=6, fill_color=secondary_color, fill_opacity=0.5, stroke_color=primary_color)
        storage_rect.next_to(process_rect, DOWN, buff=1)
        storage_label = Text("Content Storage", font_size=36).next_to(storage_rect, UP)
        storage_path = Text("data/web_content/", font_size=24, color=secondary_color).move_to(storage_rect)
        
        storage_group = VGroup(storage_rect, storage_label, storage_path)
        
        # Arrow to storage
        process_to_storage = Arrow(process_rect.get_bottom(), storage_rect.get_top(), color=primary_color)
        
        # Animate storage
        self.play(Create(process_to_storage))
        self.play(FadeIn(storage_group))
        
        self.wait(2)
        
        # Libraries used
        libraries_title = Text("Libraries Used", font_size=36, color=primary_color)
        libraries_title.next_to(storage_group, DOWN, buff=1)
        
        libraries = VGroup(
            Text("crawl4ai", font_size=24),
            Text("scrapegraphai", font_size=24),
            Text("langchain", font_size=24),
            Text("docling", font_size=24)
        ).arrange(RIGHT, buff=0.7).next_to(libraries_title, DOWN)
        
        self.play(Write(libraries_title))
        self.play(FadeIn(libraries))
        
        self.wait(2)
        
        # Clear for next section
        self.play(
            FadeOut(section_title),
            FadeOut(web_group),
            FadeOut(file_group),
            FadeOut(web_arrow),
            FadeOut(file_arrow),
            FadeOut(process_group),
            FadeOut(process_to_storage),
            FadeOut(storage_group),
            FadeOut(libraries_title),
            FadeOut(libraries)
        )
    
    def llm_integration_section(self, primary_color, highlight_color):
        # Section title
        section_title = Text("LLM Integration and RAG", font_size=56, color=primary_color).to_edge(UP)
        self.play(Write(section_title))
        self.wait(1)
        
        # LLM models support
        llm_title = Text("Supported LLMs", font_size=36, color=primary_color).shift(UP * 2)
        
        llm_models = VGroup(
            Text("OpenAI", font_size=28),
            Text("Claude", font_size=28),
            Text("Cohere", font_size=28),
            Text("DeepSeek", font_size=28),
            Text("Gemini", font_size=28),
            Text("Grok", font_size=28),
            Text("Mistral", font_size=28)
        ).arrange_in_grid(rows=2, cols=4, buff=0.8).next_to(llm_title, DOWN, buff=0.5)
        
        self.play(Write(llm_title))
        self.play(FadeIn(llm_models))
        
        self.wait(2)
        
        # RAG Process visualization
        self.play(
            FadeOut(llm_title),
            FadeOut(llm_models)
        )
        
        # RAG Components
        rag_title = Text("Retrieval-Augmented Generation (RAG)", font_size=36, color=primary_color).shift(UP * 2)
        
        # Create a visualization of RAG process
        query_box = Rectangle(height=1, width=3, fill_color=highlight_color, fill_opacity=0.3, stroke_color=primary_color)
        query_label = Text("User Query", font_size=24).next_to(query_box, UP, buff=0.2)
        query_group = VGroup(query_box, query_label).shift(LEFT * 4 + UP * 0.5)
        
        embedding_box = Rectangle(height=1.5, width=3, fill_color=primary_color, fill_opacity=0.3, stroke_color=primary_color)
        embedding_label = Text("Embedding Generation", font_size=24).next_to(embedding_box, UP, buff=0.2)
        embedding_group = VGroup(embedding_box, embedding_label).next_to(query_group, RIGHT, buff=2)
        
        vector_db = Rectangle(height=2, width=3, fill_color=highlight_color, fill_opacity=0.3, stroke_color=primary_color)
        vector_label = Text("FAISS Vector Store", font_size=24).next_to(vector_db, UP, buff=0.2)
        vector_group = VGroup(vector_db, vector_label).next_to(embedding_box, DOWN, buff=1.5)
        
        llm_box = Rectangle(height=1.5, width=3, fill_color=primary_color, fill_opacity=0.3, stroke_color=primary_color)
        llm_label = Text("LLM Processing", font_size=24).next_to(llm_box, UP, buff=0.2)
        llm_group = VGroup(llm_box, llm_label).next_to(embedding_group, RIGHT, buff=2)
        
        response_box = Rectangle(height=1, width=3, fill_color=highlight_color, fill_opacity=0.3, stroke_color=primary_color)
        response_label = Text("Response", font_size=24).next_to(response_box, UP, buff=0.2)
        response_group = VGroup(response_box, response_label).next_to(llm_group, RIGHT, buff=2)
        
        # Arrows connecting components
        query_to_embedding = Arrow(query_box.get_right(), embedding_box.get_left(), color=primary_color)
        embedding_to_vector = Arrow(embedding_box.get_bottom(), vector_db.get_top(), color=primary_color)
        vector_to_llm = Arrow(vector_db.get_right(), llm_box.get_bottom(), color=primary_color, path_arc=-np.pi/4)
        embedding_to_llm = Arrow(embedding_box.get_right(), llm_box.get_left(), color=primary_color)
        llm_to_response = Arrow(llm_box.get_right(), response_box.get_left(), color=primary_color)
        
        self.play(Write(rag_title))
        self.play(FadeIn(query_group))
        self.play(
            Create(query_to_embedding),
            FadeIn(embedding_group)
        )
        self.play(
            Create(embedding_to_vector),
            FadeIn(vector_group)
        )
        self.play(
            Create(vector_to_llm),
            Create(embedding_to_llm),
            FadeIn(llm_group)
        )
        self.play(
            Create(llm_to_response),
            FadeIn(response_group)
        )
        
        self.wait(3)
        
        # Clear for next section
        self.play(
            FadeOut(section_title),
            FadeOut(rag_title),
            FadeOut(query_group),
            FadeOut(embedding_group),
            FadeOut(vector_group),
            FadeOut(llm_group),
            FadeOut(response_group),
            FadeOut(query_to_embedding),
            FadeOut(embedding_to_vector),
            FadeOut(vector_to_llm),
            FadeOut(embedding_to_llm),
            FadeOut(llm_to_response)
        )
    
    def monitoring_section(self, primary_color, accent_color):
        # Section title
        section_title = Text("Monitoring and Uncertainty Detection", font_size=56, color=primary_color).to_edge(UP)
        self.play(Write(section_title))
        self.wait(1)
        
        # Create a flowchart of the monitoring process
        # Response box
        response_box = Rectangle(height=1.2, width=3, fill_color=primary_color, fill_opacity=0.3, stroke_color=primary_color)
        response_label = Text("LLM Response", font_size=24).move_to(response_box)
        response_group = VGroup(response_box, response_label).shift(UP * 1.5)
        
        # Classifier box
        classifier_box = Rectangle(height=1.5, width=3.5, fill_color=accent_color, fill_opacity=0.3, stroke_color=primary_color)
        classifier_label = Text("Uncertainty Classifier", font_size=24).move_to(classifier_box)
        classifier_info = Text("92.7% accuracy", font_size=18, color=accent_color).next_to(classifier_box, DOWN, buff=0.2)
        classifier_group = VGroup(classifier_box, classifier_label, classifier_info).next_to(response_group, DOWN, buff=1)
        
        # Decision diamond
        decision = Polygon(
            np.array([0, 0.75, 0]),
            np.array([1, 0, 0]),
            np.array([0, -0.75, 0]),
            np.array([-1, 0, 0]),
            fill_color=primary_color,
            fill_opacity=0.3,
            stroke_color=primary_color
        )
        decision_text = Text("Uncertain?", font_size=20).move_to(decision)
        decision_group = VGroup(decision, decision_text).next_to(classifier_group, DOWN, buff=1)
        
        # Yes/No paths
        email_box = Rectangle(height=1.2, width=3, fill_color=accent_color, fill_opacity=0.3, stroke_color=primary_color)
        email_label = Text("Email Alert", font_size=24).move_to(email_box)
        email_group = VGroup(email_box, email_label).next_to(decision, LEFT, buff=2)
        
        continue_box = Rectangle(height=1.2, width=3, fill_color=primary_color, fill_opacity=0.3, stroke_color=primary_color)
        continue_label = Text("Continue", font_size=24).move_to(continue_box)
        continue_group = VGroup(continue_box, continue_label).next_to(decision, RIGHT, buff=2)
        
        # Arrows
        response_to_classifier = Arrow(response_box.get_bottom(), classifier_box.get_top(), color=primary_color)
        classifier_to_decision = Arrow(classifier_box.get_bottom(), decision.get_top(), color=primary_color)
        decision_to_email = Arrow(decision.get_left(), email_box.get_right(), color=accent_color)
        decision_to_continue = Arrow(decision.get_right(), continue_box.get_left(), color=primary_color)
        
        # Yes/No labels
        yes_label = Text("Yes", font_size=20, color=accent_color).next_to(decision_to_email, UP, buff=0.2)
        no_label = Text("No", font_size=20, color=primary_color).next_to(decision_to_continue, UP, buff=0.2)
        
        # Play animations
        self.play(FadeIn(response_group))
        self.play(
            Create(response_to_classifier),
            FadeIn(classifier_group)
        )
        self.play(
            Create(classifier_to_decision),
            FadeIn(decision_group)
        )
        self.play(
            Create(decision_to_email),
            Create(decision_to_continue),
            Write(yes_label),
            Write(no_label)
        )
        self.play(
            FadeIn(email_group),
            FadeIn(continue_group)
        )
        
        self.wait(2)
        
        # Additional monitoring features
        monitor_title = Text("Additional Monitoring Features", font_size=36, color=primary_color)
        monitor_title.next_to(VGroup(email_group, continue_group), DOWN, buff=1.5)
        
        features = VGroup(
            Text("• Thread-based monitoring service", font_size=24),
            Text("• Periodic chat history emails", font_size=24),
            Text("• Training data collection", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(monitor_title, DOWN, buff=0.5)
        
        self.play(Write(monitor_title))
        self.play(Write(features))
        
        self.wait(2)
        
        # Clear for next section
        self.play(
            FadeOut(section_title),
            FadeOut(response_group),
            FadeOut(classifier_group),
            FadeOut(decision_group),
            FadeOut(email_group),
            FadeOut(continue_group),
            FadeOut(response_to_classifier),
            FadeOut(classifier_to_decision),
            FadeOut(decision_to_email),
            FadeOut(decision_to_continue),
            FadeOut(yes_label),
            FadeOut(no_label),
            FadeOut(monitor_title),
            FadeOut(features)
        )
    
    def chat_interfaces_section(self, primary_color, secondary_color):
        # Section title
        section_title = Text("Chat Interfaces", font_size=56, color=primary_color).to_edge(UP)
        self.play(Write(section_title))
        self.wait(1)
        
        # Create visualizations for different interfaces
        # WhatsApp Interface
        whatsapp_rect = RoundedRectangle(height=4, width=2.5, corner_radius=0.2, fill_color=secondary_color, fill_opacity=0.3, stroke_color=primary_color)
        whatsapp_logo = Text("WhatsApp", font_size=24, color=secondary_color).next_to(whatsapp_rect, UP)
        whatsapp_info = Text("Using Twilio API", font_size=18).move_to(whatsapp_rect)
        whatsapp_group = VGroup(whatsapp_rect, whatsapp_logo, whatsapp_info).shift(LEFT * 3.5)
        
        # Telegram Interface
        telegram_rect = RoundedRectangle(height=4, width=2.5, corner_radius=0.2, fill_color=secondary_color, fill_opacity=0.3, stroke_color=primary_color)
        telegram_logo = Text("Telegram", font_size=24, color=secondary_color).next_to(telegram_rect, UP)
        telegram_info = Text("python-telegram-bot", font_size=18).move_to(telegram_rect)
        telegram_group = VGroup(telegram_rect, telegram_logo, telegram_info)
        
        # Iframe Interface
        iframe_rect = RoundedRectangle(height=4, width=2.5, corner_radius=0.2, fill_color=secondary_color, fill_opacity=0.3, stroke_color=primary_color)
        iframe_logo = Text("Iframe", font_size=24, color=secondary_color).next_to(iframe_rect, UP)
        iframe_info = Text("FastAPI + HTML/CSS/JS", font_size=18).move_to(iframe_rect)
        iframe_group = VGroup(iframe_rect, iframe_logo, iframe_info).shift(RIGHT * 3.5)
        
        # Group all interfaces
        interfaces = VGroup(whatsapp_group, telegram_group, iframe_group).arrange(RIGHT, buff=1)
        
        # Play animations
        self.play(FadeIn(interfaces))
        
        self.wait(2)
        
        # Central DataVerse ChatBot
        bot_circle = Circle(radius=1.5, fill_color=primary_color, fill_opacity=0.5, stroke_color=primary_color)
        bot_text = Text("DataVerse\nChatBot", font_size=24).move_to(bot_circle)
        bot_group = VGroup(bot_circle, bot_text).next_to(interfaces, DOWN, buff=1.5)
        
        # Arrows from interfaces to bot
        whatsapp_to_bot = Arrow(whatsapp_rect.get_bottom(), bot_circle.get_top() + LEFT, color=primary_color)
        telegram_to_bot = Arrow(telegram_rect.get_bottom(), bot_circle.get_top(), color=primary_color)
        iframe_to_bot = Arrow(iframe_rect.get_bottom(), bot_circle.get_top() + RIGHT, color=primary_color)
        
        self.play(FadeIn(bot_group))
        self.play(
            Create(whatsapp_to_bot),
            Create(telegram_to_bot),
            Create(iframe_to_bot)
        )
        
        # Features text
        features_title = Text("Interface Features", font_size=36, color=primary_color)
        features_title.next_to(bot_group, DOWN, buff=1)
        
        features = VGroup(
            Text("• Multi-platform support", font_size=24),
            Text("• Voice message processing", font_size=24),
            Text("• Chat history persistence", font_size=24),
            Text("• Shared backend logic", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(features_title, DOWN, buff=0.5)
        
        self.play(Write(features_title))
        self.play(Write(features))
        
        self.wait(2)
        
        # Clear for next section
        self.play(
            FadeOut(section_title),
            FadeOut(interfaces),
            FadeOut(bot_group),
            FadeOut(whatsapp_to_bot),
            FadeOut(telegram_to_bot),
            FadeOut(iframe_to_bot),
            FadeOut(features_title),
            FadeOut(features)
        )
    
    def dataset_training_section(self, primary_color, highlight_color):
        # Section title
        section_title = Text("Dataset Creation and Model Training", font_size=56, color=primary_color).to_edge(UP)
        self.play(Write(section_title))
        self.wait(1)
        
        # Dataset creation process
        dataset_title = Text("Dataset Creation Process", font_size=36, color=primary_color).shift(UP * 2)
        
        # Flow chart for dataset creation
        rag_box = Rectangle(height=1.2, width=3, fill_color=highlight_color, fill_opacity=0.3, stroke_color=primary_color)
        rag_label = Text("RAG Responses", font_size=24).move_to(rag_box)
        rag_group = VGroup(rag_box, rag_label).shift(LEFT * 3 + UP * 0.5)
        
        clean_box = Rectangle(height=1.2, width=3, fill_color=primary_color, fill_opacity=0.3, stroke_color=primary_color)
        clean_label = Text("Data Cleaning", font_size=24).move_to(clean_box)
        clean_group = VGroup(clean_box, clean_label).next_to(rag_group, RIGHT, buff=2)
        
        tokenize_box = Rectangle(height=1.2, width=3, fill_color=primary_color, fill_opacity=0.3, stroke_color=primary_color)
        tokenize_label = Text("Tokenization", font_size=24).move_to(tokenize_box)
        tokenize_info = Text("sentence-transformers/all-MiniLM-L6-v2", font_size=18, color=highlight_color).next_to(tokenize_box, DOWN, buff=0.2)
        tokenize_group = VGroup(tokenize_box, tokenize_label, tokenize_info).next_to(clean_group, RIGHT, buff=2)
        
        dataset_box = Rectangle(height=1.2, width=3, fill_color=highlight_color, fill_opacity=0.3, stroke_color=primary_color)
        dataset_label = Text("Final Dataset", font_size=24).move_to(dataset_box)
        dataset_path = Text("data/datasets/", font_size=18, color=highlight_color).next_to(dataset_box, DOWN, buff=0.2)
        dataset_group = VGroup(dataset_box, dataset_label, dataset_path).next_to(VGroup(rag_group, clean_group, tokenize_group), DOWN, buff=2)
        
        # Arrows
        rag_to_clean = Arrow(rag_box.get_right(), clean_box.get_left(), color=primary_color)
        clean_to_tokenize = Arrow(clean_box.get_right(), tokenize_box.get_left(), color=primary_color)
        tokenize_to_dataset = CurvedArrow(
            tokenize_box.get_bottom(),
            dataset_box.get_top(),
            angle=-TAU/4,
            color=primary_color
        )
        
        self.play(Write(dataset_title))
        self.play(FadeIn(rag_group))
        self.play(
            Create(rag_to_clean),
            FadeIn(clean_group)
        )
        self.play(
            Create(clean_to_tokenize),
            FadeIn(tokenize_group)
        )
        self.play(
            Create(tokenize_to_dataset),
            FadeIn(dataset_group)
        )
        
        self.wait(2)
        
        # Clear dataset section
        self.play(
            FadeOut(dataset_title),
            FadeOut(rag_group),
            FadeOut(clean_group),
            FadeOut(tokenize_group),
            FadeOut(dataset_group),
            FadeOut(rag_to_clean),
            FadeOut(clean_to_tokenize),
            FadeOut(tokenize_to_dataset)
        )
        
        # Model training section
        training_title = Text("Model Training Process", font_size=36, color=primary_color).shift(UP * 2)
        
        # Flow chart for model training
        dataset_box2 = Rectangle(height=1.2, width=3, fill_color=highlight_color, fill_opacity=0.3, stroke_color=primary_color)
        dataset_label2 = Text("Dataset", font_size=24).move_to(dataset_box2)
        dataset_group2 = VGroup(dataset_box2, dataset_label2).shift(LEFT * 4 + UP * 0.5)
        
        models_box = Rectangle(height=2, width=3, fill_color=primary_color, fill_opacity=0.3, stroke_color=primary_color)
        models_label = Text("Models", font_size=24).next_to(models_box, UP, buff=0.2)
        models_list = VGroup(
            Text("• Random Forest", font_size=18),
            Text("• XGBoost", font_size=18)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(models_box)
        models_group = VGroup(models_box, models_label, models_list).next_to(dataset_group2, RIGHT, buff=2)
        
        tuning_box = Rectangle(height=1.5, width=3, fill_color=primary_color, fill_opacity=0.3, stroke_color=primary_color)
        tuning_label = Text("Hyperparameter Tuning", font_size=24).next_to(tuning_box, UP, buff=0.2)
        tuning_info = Text("RandomizedSearchCV", font_size=18).move_to(tuning_box)
        tuning_group = VGroup(tuning_box, tuning_label, tuning_info).next_to(models_group, RIGHT, buff=2)
        
        eval_box = Rectangle(height=2.5, width=3, fill_color=highlight_color, fill_opacity=0.3, stroke_color=primary_color)
        eval_label = Text("Evaluation", font_size=24).next_to(eval_box, UP, buff=0.2)
        eval_metrics = VGroup(
            Text("• Accuracy: 92.7%", font_size=18),
            Text("• Precision", font_size=18),
            Text("• Recall", font_size=18),
            Text("• ROC Curve", font_size=18)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(eval_box)
        eval_group = VGroup(eval_box, eval_label, eval_metrics).next_to(tuning_group, RIGHT, buff=2)
        
        final_box = Rectangle(height=1.2, width=3, fill_color=highlight_color, fill_opacity=0.3, stroke_color=primary_color)
        final_label = Text("Saved Model", font_size=24).move_to(final_box)
        final_info = Text(".pkl files with metadata", font_size=18).next_to(final_box, DOWN, buff=0.2)
        final_group = VGroup(final_box, final_label, final_info).next_to(VGroup(dataset_group2, tuning_group, eval_group), DOWN, buff=2)
        
        # Arrows
        dataset_to_models = Arrow(dataset_box2.get_right(), models_box.get_left(), color=primary_color)
        models_to_tuning = Arrow(models_box.get_right(), tuning_box.get_left(), color=primary_color)
        tuning_to_eval = Arrow(tuning_box.get_right(), eval_box.get_left(), color=primary_color)
        eval_to_final = Arrow(eval_box.get_bottom(), final_box.get_top() + RIGHT, color=primary_color)
        
        self.play(Write(training_title))
        self.play(FadeIn(dataset_group2))
        self.play(
            Create(dataset_to_models),
            FadeIn(models_group)
        )
        self.play(
            Create(models_to_tuning),
            FadeIn(tuning_group)
        )
        self.play(
            Create(tuning_to_eval),
            FadeIn(eval_group)
        )
        self.play(
            Create(eval_to_final),
            FadeIn(final_group)
        )
        
        self.wait(3)
        
        # Clear for next section
        self.play(
            FadeOut(section_title),
            FadeOut(training_title),
            FadeOut(dataset_group2),
            FadeOut(models_group),
            FadeOut(tuning_group),
            FadeOut(eval_group),
            FadeOut(final_group),
            FadeOut(dataset_to_models),
            FadeOut(models_to_tuning),
            FadeOut(tuning_to_eval),
            FadeOut(eval_to_final)
        )
    
    def architecture_overview(self, primary_color, secondary_color, accent_color, highlight_color):
        # Section title
        section_title = Text("Architecture Overview", font_size=56, color=primary_color).to_edge(UP)
        self.play(Write(section_title))
        self.wait(1)
        
        # Create a comprehensive diagram of the entire system
        # Main components
        data_rect = Rectangle(height=2, width=3, fill_color=secondary_color, fill_opacity=0.3, stroke_color=primary_color)
        data_label = Text("Data Sources", font_size=24).move_to(data_rect)
        data_group = VGroup(data_rect, data_label).shift(LEFT * 5 + UP * 2)
        
        extraction_rect = Rectangle(height=2, width=3, fill_color=primary_color, fill_opacity=0.3, stroke_color=primary_color)
        extraction_label = Text("Data Extraction", font_size=24).move_to(extraction_rect)
        extraction_group = VGroup(extraction_rect, extraction_label).next_to(data_group, RIGHT, buff=1.5)
        
        embedding_rect = Rectangle(height=2, width=3, fill_color=primary_color, fill_opacity=0.3, stroke_color=primary_color)
        embedding_label = Text("Embedding Generation", font_size=24).move_to(embedding_rect)
        embedding_group = VGroup(embedding_rect, embedding_label).next_to(extraction_group, RIGHT, buff=1.5)
        
        storage_rect = Rectangle(height=2, width=3, fill_color=secondary_color, fill_opacity=0.3, stroke_color=primary_color)
        storage_label = Text("Vector Storage", font_size=24).move_to(storage_rect)
        storage_info = Text("FAISS", font_size=18, color=secondary_color).next_to(storage_rect, DOWN, buff=0.2)
        storage_group = VGroup(storage_rect, storage_label, storage_info).next_to(embedding_group, RIGHT, buff=1.5)
        
        # Second row
        query_rect = Rectangle(height=2, width=3, fill_color=highlight_color, fill_opacity=0.3, stroke_color=primary_color)
        query_label = Text("User Query", font_size=24).move_to(query_rect)
        query_group = VGroup(query_rect, query_label).next_to(data_group, DOWN, buff=2)
        
        rag_rect = Rectangle(height=2, width=3, fill_color=primary_color, fill_opacity=0.3, stroke_color=primary_color)
        rag_label = Text("RAG System", font_size=24).move_to(rag_rect)
        rag_group = VGroup(rag_rect, rag_label).next_to(query_group, RIGHT, buff=1.5)
        
        llm_rect = Rectangle(height=2, width=3, fill_color=primary_color, fill_opacity=0.3, stroke_color=primary_color)
        llm_label = Text("LLM Processing", font_size=24).move_to(llm_rect)
        llm_group = VGroup(llm_rect, llm_label).next_to(rag_group, RIGHT, buff=1.5)
        
        monitor_rect = Rectangle(height=2, width=3, fill_color=accent_color, fill_opacity=0.3, stroke_color=primary_color)
        monitor_label = Text("Monitoring", font_size=24).move_to(monitor_rect)
        monitor_group = VGroup(monitor_rect, monitor_label).next_to(llm_group, RIGHT, buff=1.5)
        
        # Third row
        interface_rect = Rectangle(height=2, width=12, fill_color=secondary_color, fill_opacity=0.3, stroke_color=primary_color)
        interface_label = Text("Chat Interfaces", font_size=24).move_to(interface_rect)
        interface_info = Text("WhatsApp, Telegram, Iframe", font_size=18).next_to(interface_label, DOWN, buff=0.3)
        interface_group = VGroup(interface_rect, interface_label, interface_info).next_to(VGroup(query_group, rag_group, llm_group), DOWN, buff=2)
        
        # Arrows for top row
        data_to_extraction = Arrow(data_rect.get_right(), extraction_rect.get_left(), color=primary_color)
        extraction_to_embedding = Arrow(extraction_rect.get_right(), embedding_rect.get_left(), color=primary_color)
        embedding_to_storage = Arrow(embedding_rect.get_right(), storage_rect.get_left(), color=primary_color)
        
        # Arrows for middle row
        query_to_rag = Arrow(query_rect.get_right(), rag_rect.get_left(), color=primary_color)
        rag_to_llm = Arrow(rag_rect.get_right(), llm_rect.get_left(), color=primary_color)
        llm_to_monitor = Arrow(llm_rect.get_right(), monitor_rect.get_left(), color=primary_color)
        
        # Cross-connections
        storage_to_rag = Arrow(storage_rect.get_bottom(), rag_rect.get_top() + RIGHT * 1.5, color=primary_color, stroke_width=2)
        
        # Arrows to interfaces
        rag_to_interface = Arrow(rag_rect.get_bottom(), interface_rect.get_top() + LEFT * 2, color=primary_color)
        llm_to_interface = Arrow(llm_rect.get_bottom(), interface_rect.get_top(), color=primary_color)
        monitor_to_interface = Arrow(monitor_rect.get_bottom(), interface_rect.get_top() + RIGHT * 2, color=primary_color)
        
        # Play animations in sequence
        # First row
        self.play(FadeIn(data_group))
        self.play(
            Create(data_to_extraction),
            FadeIn(extraction_group)
        )
        self.play(
            Create(extraction_to_embedding),
            FadeIn(embedding_group)
        )
        self.play(
            Create(embedding_to_storage),
            FadeIn(storage_group)
        )
        
        # Second row
        self.play(FadeIn(query_group))
        self.play(
            Create(query_to_rag),
            FadeIn(rag_group)
        )
        self.play(
            Create(rag_to_llm),
            FadeIn(llm_group)
        )
        self.play(
            Create(llm_to_monitor),
            FadeIn(monitor_group)
        )
        
        # Cross-connections
        self.play(Create(storage_to_rag))
        
        # Third row
        self.play(FadeIn(interface_group))
        self.play(
            Create(rag_to_interface),
            Create(llm_to_interface),
            Create(monitor_to_interface)
        )
        
        self.wait(3)
        
        # Clear for conclusion
        self.play(
            FadeOut(section_title),
            FadeOut(data_group),
            FadeOut(extraction_group),
            FadeOut(embedding_group),
            FadeOut(storage_group),
            FadeOut(query_group),
            FadeOut(rag_group),
            FadeOut(llm_group),
            FadeOut(monitor_group),
            FadeOut(interface_group),
            FadeOut(data_to_extraction),
            FadeOut(extraction_to_embedding),
            FadeOut(embedding_to_storage),
            FadeOut(query_to_rag),
            FadeOut(rag_to_llm),
            FadeOut(llm_to_monitor),
            FadeOut(storage_to_rag),
            FadeOut(rag_to_interface),
            FadeOut(llm_to_interface),
            FadeOut(monitor_to_interface)
        )
    
    def conclusion(self, primary_color):
        # Final title
        title = Text("DataVerse ChatBot", font_size=72, color=primary_color).to_edge(UP)
        
        # Key benefits
        benefits_title = Text("Key Benefits", font_size=48, color=primary_color).next_to(title, DOWN, buff=1)
        
        benefits = VGroup(
            Text("• Multi-source data extraction and processing", font_size=32),
            Text("• Integration with leading LLMs", font_size=32),
            Text("• Advanced monitoring with uncertainty detection", font_size=32),
            Text("• Multiple chat interfaces", font_size=32),
            Text("• Modular, extensible architecture", font_size=32)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(benefits_title, DOWN, buff=0.8)
        
        self.play(Write(title))
        self.play(Write(benefits_title))
        self.play(Write(benefits))
        
        self.wait(3)
        
        # Final fade out
        self.play(
            FadeOut(title),
            FadeOut(benefits_title),
            FadeOut(benefits)
        )

# Command to render the animation
if __name__ == "__main__":
    # Use the following command to render:
    # manim -p -qh dataverse_animation.py DataVerseChatBot
    scene = DataVerseChatBot()
    scene.render()