import 'dart:io';
import 'dart:math';

import 'package:material_ui/material_ui.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'api.dart';

void main() => runApp(const PhotoAdvisor());

class PhotoAdvisor extends StatelessWidget {
  const PhotoAdvisor({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'Photo Advisor',
    theme: ThemeData(
      colorSchemeSeed: const Color(0xff236555),
      useMaterial3: true,
    ),
    home: const Studio(),
  );
}

class Studio extends StatefulWidget {
  const Studio({super.key});
  @override
  State<Studio> createState() => _StudioState();
}

class _StudioState extends State<Studio> {
  static const goals = {
    'natural_portrait': 'Chân dung tự nhiên',
    'vibrant': 'Màu rực rỡ',
    'silhouette': 'Bóng ngược sáng',
    'full_body': 'Toàn thân',
    'keep_context': 'Giữ bối cảnh',
    'memorial': 'Ảnh kỷ niệm',
  };
  final api = PhotoApi(
    Uri.parse(
      const String.fromEnvironment(
        'API_URL',
        defaultValue: 'http://127.0.0.1:8000',
      ),
    ),
  );
  final reason = TextEditingController();
  final session =
      'session_${DateTime.now().microsecondsSinceEpoch}_${Random.secure().nextInt(1 << 30)}';
  late final historyKey = 'jobs_${api.base}';
  List<String> history = [];
  XFile? file;
  String goal = 'natural_portrait';
  bool consent = false, busy = false;
  String? job, selection, message;
  bool? preference;
  List<Map<String, dynamic>> options = [];

  @override
  void initState() {
    super.initState();
    run(() async {
      history =
          (await SharedPreferences.getInstance()).getStringList(historyKey) ??
          [];
    });
  }

  @override
  void dispose() {
    reason.dispose();
    super.dispose();
  }

  Future<void> run(Future<void> Function() action) async {
    if (busy) return;
    setState(() {
      busy = true;
      message = null;
    });
    try {
      await action();
    } catch (e) {
      message = e.toString();
    } finally {
      if (mounted) setState(() => busy = false);
    }
  }

  Future<void> load(String id) async {
    final result = await api.candidates(id);
    options = result..shuffle();
    job = id;
    selection = null;
    preference = null;
    reason.clear();
  }

  Future<void> saveHistory() async {
    await (await SharedPreferences.getInstance()).setStringList(
      historyKey,
      history,
    );
  }

  Future<void> submit() async {
    final id = await api.upload(file!, goal);
    history = [id, ...history];
    await saveHistory();
    await load(id);
  }

  Future<void> pick(ImageSource source) async {
    final photo = await ImagePicker().pickImage(source: source);
    if (photo != null) {
      file = photo;
      consent = false;
      job = null;
      options = [];
      selection = null;
    }
  }

  @override
  Widget build(BuildContext context) {
    final selected = options
        .where((c) => c['candidate_id'] == selection)
        .firstOrNull;
    return Scaffold(
      appBar: AppBar(title: const Text('Photo Advisor')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            Wrap(
              spacing: 12,
              children: [
                OutlinedButton.icon(
                  onPressed: busy
                      ? null
                      : () => run(() => pick(ImageSource.gallery)),
                  icon: const Icon(Icons.photo_library_outlined),
                  label: const Text('Chọn ảnh'),
                ),
                OutlinedButton.icon(
                  onPressed: busy
                      ? null
                      : () => run(() => pick(ImageSource.camera)),
                  icon: const Icon(Icons.camera_alt_outlined),
                  label: const Text('Chụp ảnh'),
                ),
              ],
            ),
            if (file != null) Text(file!.name),
            DropdownButton<String>(
              value: goal,
              isExpanded: true,
              items: goals.entries
                  .map(
                    (e) => DropdownMenuItem(value: e.key, child: Text(e.value)),
                  )
                  .toList(),
              onChanged: busy ? null : (v) => setState(() => goal = v!),
            ),
            CheckboxListTile(
              contentPadding: EdgeInsets.zero,
              value: consent,
              onChanged: busy ? null : (v) => setState(() => consent = v!),
              title: const Text(
                'Tôi đồng ý gửi ảnh đến máy chủ để xử lý và lưu kết quả.',
              ),
            ),
            FilledButton(
              onPressed: busy || file == null || !consent
                  ? null
                  : () => run(submit),
              child: const Text('Tạo phương án'),
            ),
            if (busy) const LinearProgressIndicator(),
            if (message != null) SelectableText(message!),
            for (final c in options)
              Card(
                child: InkWell(
                  onTap: busy
                      ? null
                      : () => setState(() {
                          selection = c['candidate_id'];
                          preference = null;
                        }),
                  child: Column(
                    children: [
                      Image.network(
                        api.url(c['preview_url']).toString(),
                        height: 190,
                        cacheHeight: 500,
                        fit: BoxFit.contain,
                        errorBuilder: (ctx, error, stack) =>
                            const Text('Không tải được ảnh xem trước.'),
                      ),
                      ListTile(
                        title: Text(c['style_label'] ?? c['name']),
                        trailing: Icon(
                          selection == c['candidate_id']
                              ? Icons.check_circle
                              : Icons.circle_outlined,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            if (selected != null) ...[
              DropdownButton<bool>(
                value: preference,
                hint: const Text('So sánh với ảnh gốc (tùy chọn)'),
                items: const [
                  DropdownMenuItem(
                    value: true,
                    child: Text('Tôi thích phương án đã chọn hơn'),
                  ),
                  DropdownMenuItem(
                    value: false,
                    child: Text('Tôi không thích hơn ảnh gốc'),
                  ),
                ],
                onChanged: busy ? null : (v) => setState(() => preference = v),
              ),
              TextField(
                controller: reason,
                maxLength: 500,
                decoration: const InputDecoration(
                  labelText: 'Lý do chọn (tùy chọn)',
                ),
              ),
              FilledButton(
                onPressed: busy
                    ? null
                    : () => run(() async {
                        await api.feedback(
                          job!,
                          session,
                          selection!,
                          options
                              .map((c) => c['candidate_id'] as String)
                              .toList(),
                          preference,
                          reason.text,
                        );
                        message = 'Đã ghi nhận phản hồi.';
                      }),
                child: const Text('Gửi phản hồi'),
              ),
              OutlinedButton(
                onPressed: busy || selected['export_eligible'] != true
                    ? null
                    : () => run(() async {
                        final data = await api.export(job!, selection!);
                        final target = File(
                          '${(await getApplicationDocumentsDirectory()).path}/$selection.jpg',
                        );
                        await target.writeAsBytes(data);
                        message = 'Đã lưu JPEG tại ${target.path}';
                      }),
                child: const Text('Lưu JPEG'),
              ),
            ],
            if (history.isNotEmpty)
              const ListTile(title: Text('Lịch sử trên thiết bị')),
            for (final id in history)
              ListTile(
                title: Text(id),
                onTap: busy ? null : () => run(() => load(id)),
                trailing: IconButton(
                  tooltip: 'Xóa dữ liệu trên máy chủ',
                  icon: const Icon(Icons.delete_outline),
                  onPressed: busy
                      ? null
                      : () => run(() async {
                          final confirmed = await showDialog<bool>(
                            context: context,
                            builder: (ctx) => AlertDialog(
                              title: const Text(
                                'Xóa ảnh và phản hồi trên máy chủ?',
                              ),
                              actions: [
                                TextButton(
                                  onPressed: () => Navigator.pop(ctx, false),
                                  child: const Text('Hủy'),
                                ),
                                TextButton(
                                  onPressed: () => Navigator.pop(ctx, true),
                                  child: const Text('Xóa'),
                                ),
                              ],
                            ),
                          );
                          if (confirmed != true) return;
                          await api.delete(id);
                          history.remove(id);
                          await saveHistory();
                          if (job == id) {
                            job = null;
                            options = [];
                            selection = null;
                          }
                        }),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
