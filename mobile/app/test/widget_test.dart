import 'package:material_ui/material_ui.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:photo_advisor/main.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  testWidgets('upload requires a selected image and consent', (tester) async {
    SharedPreferences.setMockInitialValues({});
    await tester.pumpWidget(const PhotoAdvisor());
    await tester.pumpAndSettle();
    expect(find.text('Chọn ảnh'), findsOneWidget);
    expect(find.text('Chụp ảnh'), findsOneWidget);
    expect(
      tester.widget<FilledButton>(find.byType(FilledButton)).onPressed,
      isNull,
    );
    await tester.tap(find.byType(CheckboxListTile));
    await tester.pump();
    expect(
      tester.widget<FilledButton>(find.byType(FilledButton)).onPressed,
      isNull,
    );
  });
}
