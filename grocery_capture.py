import os
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2

class GroceryCaptureApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')

        # Status label
        self.status_label = Label(text="Press 'Space' to capture image", size_hint_y=None, height=50)
        self.layout.add_widget(self.status_label)

        # Image widget
        self.img = Image(size_hint=(1, 0.9))
        self.layout.add_widget(self.img)

        # Initialize camera
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.status_label.text = "Error: Camera could not be opened."
            return self.layout

        # Start camera feed
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        from kivy.core.window import Window
        Window.bind(on_key_down=self.on_key_down)

        return self.layout

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.img.texture = self.create_texture(frame)

    def create_texture(self, frame):
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(frame.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        texture.flip_vertical()
        return texture

    def on_key_down(self, window, key, *args):
        if key == 32:  # Space key
            self.capture_image()

    def capture_image(self):
        ret, frame = self.capture.read()
        if ret:
            uploads_directory = 'uploads'
            os.makedirs(uploads_directory, exist_ok=True)
            image_path = os.path.join(uploads_directory, "captured_image.jpg")
            cv2.imwrite(image_path, frame)  # Save the image
            self.status_label.text = "Image captured and saved successfully!"
            self.display_captured_image(frame)
        else:
            self.status_label.text = "Error: Failed to capture image."

    def display_captured_image(self, frame):
        captured_img = Image(size_hint=(1, 1), allow_stretch=True)
        captured_img.texture = self.create_texture(frame)
        self.show_popup(captured_img)

    def show_popup(self, content):
        close_button = Button(text='Close', size_hint_y=None, height=50)
        close_button.bind(on_release=self.close_popup)
        popup_content = BoxLayout(orientation='vertical')
        popup_content.add_widget(content)
        popup_content.add_widget(close_button)
        self.popup = Popup(title='Captured Image', content=popup_content, size_hint=(0.8, 0.8), auto_dismiss=False)
        self.popup.open()

    def close_popup(self, instance):
        self.popup.dismiss()

    def on_stop(self):
        if self.capture.isOpened():
            self.capture.release()

if __name__ == '__main__':
    GroceryCaptureApp().run()
