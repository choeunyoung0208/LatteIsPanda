#include <ESP8266WiFi.h>
#include <Adafruit_NeoPixel.h> // LED
#include <SPI.h>  // LED

/*
const char* ssid = "317-4";
const char* password = "lemon3174";
*/

// 연결할 wifi 이름과 비밀번호를 설정한다
const char* ssid = "1212";
const char* password = "123456789a";

// LED 
#define PIN            14   // GPIO 14(NodeMCU)
#define STRIPSIZE      46    // LED 개수
Adafruit_NeoPixel strip = Adafruit_NeoPixel(STRIPSIZE, PIN, NEO_GRB + NEO_KHZ800);
////


WiFiServer server(12345);// 포트번호를 설정한다

void setup() {
  // LED를 설정한다
  strip.begin();
  strip.setBrightness(120);
  strip.show();
  ////
  
  Serial.begin(115200); // 시리얼통신을 설정한다
  delay(10);
 
  // 연결할 wifi를 출력한다
  Serial.println();
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);// 미리 설정된 wifi에 연결을 시도한다

  // wifi연결을 대기하는 동안 ...을 출력한다
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  // wifi가 연결되었다고 출력한다
  Serial.println("");
  Serial.println("WiFi connected");

  // server역할을 시작한다
  server.begin();
  Serial.println("Server started");

  // 현재 ip를 출력한다
  Serial.print("Use this URL to connect: ");
  Serial.print("http://");
  Serial.print(WiFi.localIP());
  Serial.println("/");
  ////

  rgb(0,255,0); // LED의 색상을 초록색으로 초기화한다
}

void loop() {
  
  WiFiClient client=server.available(); // 연결을 요청한 client가 있는지 확인한다

  if (!client) { //client가 연결되지 않았다면
    Serial.println("connection waiting"); //서버 접속에 실패했음을 출력한다
    return;
  }
  else
  {
    while(client.connected()){ // client가 연결되었다면
      
      if(client.available()){
         char recevbline= client.read(); // client가 전송한 data를 읽는다
         Serial.println(recevbline);
         if(recevbline=='0') { // '0'을 전송받은 경우
          rgb(255,0,0); // LED를 빨간색으로 설정한다
         } 
         else { // '1'을 전송받은 경우
          rgb(0,255,0); // LED를 초록색으로 설정한다
         }
      }
    }
  }
}



// LED를 출력하는 함수
void rgb(int r,int g, int b){
  for(int i = 0;i<=46;i++){
    strip.setPixelColor(i,r,g,b);
  } 
  strip.show();
}
