// Positive test cases for Gemini (Dart)

// ruleid: detect-gemini
import 'package:google_generative_ai/google_generative_ai.dart';

void main() async {
  // ruleid: detect-gemini
  final model = GenerativeModel(model: 'gemini-pro', apiKey: 'key');

  // ruleid: detect-gemini
  final response = await model.generateContent([Content.text('Hello')]);

  // ruleid: detect-gemini
  final chat = model.startChat();
}
