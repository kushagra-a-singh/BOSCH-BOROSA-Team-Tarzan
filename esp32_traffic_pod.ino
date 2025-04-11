#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// WiFi credentials
const char *ssid = "YOUR_WIFI_SSID";
const char *password = "YOUR_WIFI_PASSWORD";

// MQTT Broker settings
const char *mqtt_server = "YOUR_MQTT_BROKER_IP";
const int mqtt_port = 1883;

// Pin definitions
const int MOTOR_LEFT_IN1 = 12;
const int MOTOR_LEFT_IN2 = 13;
const int MOTOR_RIGHT_IN1 = 14;
const int MOTOR_RIGHT_IN2 = 15;
const int BUZZER_PIN = 16;

// PWM channels
const int PWM_CHANNEL_LEFT = 0;
const int PWM_CHANNEL_RIGHT = 1;
const int PWM_FREQ = 5000;
const int PWM_RESOLUTION = 8;

WiFiClient espClient;
PubSubClient client(espClient);

// Current state variables
int currentSpeedLeft = 0;
int currentSpeedRight = 0;
bool buzzerActive = false;

void setup()
{
    Serial.begin(115200);

    // Setup motor pins
    pinMode(MOTOR_LEFT_IN1, OUTPUT);
    pinMode(MOTOR_LEFT_IN2, OUTPUT);
    pinMode(MOTOR_RIGHT_IN1, OUTPUT);
    pinMode(MOTOR_RIGHT_IN2, OUTPUT);
    pinMode(BUZZER_PIN, OUTPUT);

    // Configure PWM
    ledcSetup(PWM_CHANNEL_LEFT, PWM_FREQ, PWM_RESOLUTION);
    ledcSetup(PWM_CHANNEL_RIGHT, PWM_FREQ, PWM_RESOLUTION);
    ledcAttachPin(MOTOR_LEFT_IN1, PWM_CHANNEL_LEFT);
    ledcAttachPin(MOTOR_RIGHT_IN1, PWM_CHANNEL_RIGHT);

    setup_wifi();
    client.setServer(mqtt_server, mqtt_port);
    client.setCallback(callback);
}

void setup_wifi()
{
    delay(10);
    Serial.println("Connecting to WiFi...");

    WiFi.begin(ssid, password);

    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }

    Serial.println("\nWiFi connected");
    Serial.println("IP address: ");
    Serial.println(WiFi.localIP());
}

void callback(char *topic, byte *payload, unsigned int length)
{
    StaticJsonDocument<200> doc;

    // Convert payload to string
    char message[length + 1];
    memcpy(message, payload, length);
    message[length] = '\0';

    DeserializationError error = deserializeJson(doc, message);

    if (error)
    {
        Serial.println("Failed to parse JSON");
        return;
    }

    // Handle state changes
    const char *state = doc["state"];

    if (doc.containsKey("buzzer"))
    {
        buzzerActive = doc["buzzer"];
        digitalWrite(BUZZER_PIN, buzzerActive ? HIGH : LOW);
    }

    // Handle speed values if present
    if (doc.containsKey("speed_left"))
    {
        currentSpeedLeft = doc["speed_left"];
    }
    if (doc.containsKey("speed_right"))
    {
        currentSpeedRight = doc["speed_right"];
    }

    // Update motor speeds based on state
    if (strcmp(state, "go") == 0)
    {
        setMotorSpeeds(255, 255);
    }
    else if (strcmp(state, "slow") == 0)
    {
        setMotorSpeeds(128, 128);
    }
    else if (strcmp(state, "halt") == 0)
    {
        setMotorSpeeds(0, 0);
    }
    else if (strcmp(state, "go_slow") == 0)
    {
        setMotorSpeeds(currentSpeedLeft, currentSpeedRight);
    }
    else if (strcmp(state, "slow_go") == 0)
    {
        setMotorSpeeds(currentSpeedLeft, currentSpeedRight);
    }
}

void setMotorSpeeds(int leftSpeed, int rightSpeed)
{
    // Set left motor
    if (leftSpeed >= 0)
    {
        ledcWrite(PWM_CHANNEL_LEFT, leftSpeed);
        digitalWrite(MOTOR_LEFT_IN2, LOW);
    }
    else
    {
        ledcWrite(PWM_CHANNEL_LEFT, -leftSpeed);
        digitalWrite(MOTOR_LEFT_IN2, HIGH);
    }

    // Set right motor
    if (rightSpeed >= 0)
    {
        ledcWrite(PWM_CHANNEL_RIGHT, rightSpeed);
        digitalWrite(MOTOR_RIGHT_IN2, LOW);
    }
    else
    {
        ledcWrite(PWM_CHANNEL_RIGHT, -rightSpeed);
        digitalWrite(MOTOR_RIGHT_IN2, HIGH);
    }
}

void reconnect()
{
    while (!client.connected())
    {
        Serial.println("Attempting MQTT connection...");
        if (client.connect("ESP32Client"))
        {
            Serial.println("Connected to MQTT broker");
            client.subscribe("traffic/esp32/state");
        }
        else
        {
            Serial.print("Failed, rc=");
            Serial.print(client.state());
            Serial.println(" Retrying in 5 seconds...");
            delay(5000);
        }
    }
}

void loop()
{
    if (!client.connected())
    {
        reconnect();
    }
    client.loop();
}