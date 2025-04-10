#include "esp_camera.h"
#include <WiFi.h>
#include <PubSubClient.h>
#include "driver/ledc.h"
#include <esp32-hal-ledc.h>

#define CAMERA_MODEL_XIAO_ESP32S3
#include "camera_pins.h"

const char* ssid = "taksh";
const char* password = "12345678";
const char* mqtt_server = "broker.hivemq.com";  // You can use your own broker

WiFiClient espClient;
PubSubClient client(espClient);

#define ENA 1
#define IN1 2
#define IN2 3
#define IN3 4
#define IN4 5
#define ENB 6

int motorSpeed = 200;

void stopMotors() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
  ledcWrite(0, 0); ledcWrite(1, 0);
}

void moveForward() {
  stopMotors();
  delay(100);
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
  ledcWrite(0, motorSpeed); ledcWrite(1, motorSpeed);
}

void moveBackward() {
  stopMotors();
  delay(100);
  digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
  ledcWrite(0, motorSpeed); ledcWrite(1, motorSpeed);
}

void setup_wifi() {
  delay(10);
  Serial.println("🔌 Connecting to Wi-Fi");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print(".");
  }
  Serial.println("\n✅ Wi-Fi connected");
  Serial.println(WiFi.localIP());
}

void callback(char* topic, byte* payload, unsigned int length) {
  String command;
  for (unsigned int i = 0; i < length; i++) command += (char)payload[i];
  command.trim();

  Serial.print("📥 Command received: ");
  Serial.println(command);

  if (command == "forward") moveForward();
  else if (command == "backward") moveBackward();
  else if (command == "stop") stopMotors();
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("🔁 Attempting MQTT connection...");
    if (client.connect("ESP32CamClient")) {
      Serial.println("✅ Connected");
      client.subscribe("esp32/cam/control");
    } else {
      Serial.print("❌ Failed, rc=");
      Serial.print(client.state());
      delay(2000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println("🚗 Starting ESP32-S3 Camera + MQTT Control");

  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  ledcSetup(0, 1000, 8); ledcSetup(1, 1000, 8);
  ledcAttachPin(ENA, 0); ledcAttachPin(ENB, 1);
  stopMotors();

  setup_wifi();
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 10;
  config.fb_count = 2;
  config.frame_size = FRAMESIZE_QVGA;

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("❌ Camera init failed");
    return;
  }
}

void loop() {
  if (!client.connected()) reconnect();
  client.loop();

  // Optional auto-movement
  // moveForward(); delay(2000);
  // moveBackward(); delay(2000);
  // stopMotors(); delay(1000);
}
