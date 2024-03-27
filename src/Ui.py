from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.spinner import Spinner
from kivy.uix.dropdown import DropDown
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from prediction import predict_image, load_model



class PredictScreen(Screen):
    def __init__(self, **kwargs):
        super(PredictScreen, self).__init__(**kwargs)
        self.file_chooser = FileChooserIconView(path='.', filters=['*.jpg', '*.png'])
        self.add_widget(self.file_chooser)
        self.predict_button = Button(text='Predict', on_press=self.predict)
        self.add_widget(self.predict_button)
        self.result_label = Label(text='')
        self.add_widget(self.result_label)

    def predict(self, instance):
        if len(self.file_chooser.selection) == 0:
            popup = Popup(title='Error', content=Label(text='No image selected'), size_hint=(None, None), size=(400, 400))
            popup.open()
        else:
            image_path = self.file_chooser.selection[0]
            result = predict_image(image_path, load_model())
            self.result_label.text = f'Prediction: {result}'

class MyApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(PredictScreen(name='predict'))
        return sm

if __name__ == '__main__':
    MyApp().run()
