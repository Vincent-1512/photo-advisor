import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

class PhotoApi {
  PhotoApi(this.base);
  final Uri base;
  static const timeout = Duration(minutes: 2);

  Uri url(String path) => base.resolve(path);

  Future<http.Response> send(http.BaseRequest request) async {
    final client = http.Client();
    try {
      final response = await client
          .send(request)
          .then(http.Response.fromStream)
          .timeout(timeout);
      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw StateError(
          'API ${response.statusCode}: ${utf8.decode(response.bodyBytes, allowMalformed: true)}',
        );
      }
      return response;
    } finally {
      client.close();
    }
  }

  Future<String> upload(XFile image, String goal) async {
    if (await image.length() > 15 * 1024 * 1024) {
      throw const FormatException('Ảnh vượt giới hạn 15 MiB.');
    }
    final bytes = await image.readAsBytes();
    final isPng = bytes.length >= 8 && bytes[0] == 137 && bytes[1] == 80;
    final isJpeg = bytes.length >= 2 && bytes[0] == 255 && bytes[1] == 216;
    if (!isPng && !isJpeg) {
      throw const FormatException('Hãy chọn ảnh JPEG hoặc PNG.');
    }
    final request = http.MultipartRequest('POST', url('/v1/jobs'))
      ..fields.addAll({'goal': goal, 'consent': 'true'})
      ..files.add(
        http.MultipartFile.fromBytes(
          'image',
          bytes,
          filename: isPng ? 'photo.png' : 'photo.jpg',
          contentType: http.MediaType('image', isPng ? 'png' : 'jpeg'),
        ),
      );
    return jsonDecode((await send(request)).body)['job_id'] as String;
  }

  Future<List<Map<String, dynamic>>> candidates(String job) async {
    for (var attempt = 0; attempt < 60; attempt++) {
      final response = await send(http.Request('GET', url('/v1/jobs/$job')));
      final data = jsonDecode(utf8.decode(response.bodyBytes));
      if (data['status'] == 'succeeded') {
        return List<Map<String, dynamic>>.from(data['result']['candidates']);
      }
      if (data['status'] == 'failed') {
        throw StateError(data['error']['message']);
      }
      await Future<void>.delayed(const Duration(seconds: 1));
    }
    throw StateError('Hết thời gian chờ kết quả.');
  }

  Future<void> feedback(
    String job,
    String session,
    String choice,
    List<String> order,
    bool? preference,
    String reason,
  ) async {
    await send(
      http.Request('POST', url('/v1/jobs/$job/feedback'))
        ..headers['Content-Type'] = 'application/json'
        ..body = jsonEncode({
          'session_id': session,
          'chosen_candidate_id': choice,
          'shown_order': order,
          'preferred_over_original': preference,
          'reason': reason,
        }),
    );
  }

  Future<Uint8List> export(String job, String choice) async => (await send(
    http.Request('POST', url('/v1/jobs/$job/candidates/$choice/export')),
  )).bodyBytes;

  Future<void> delete(String job) async {
    await send(http.Request('DELETE', url('/v1/jobs/$job')));
  }
}
