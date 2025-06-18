import asyncio
import websockets
import time 

async def websocket_client():
    uri = "ws://localhost:8000/ws"  # Thay bằng WebSocket server của bạn
    async with websockets.connect(uri) as websocket:
        print("Connected to server")
        while True:
            time.sleep(1)
            # Gửi dữ liệu (text)
            await websocket.send("Hello server!")
            print("Sent message.")
            try:
                # Nhận dữ liệu từ server
                message = await websocket.recv()
                print(f"Received message: {message}")
            except websockets.ConnectionClosed:
                print("Connection closed by server.")
                break
        
        

        # # Nhận phản hồi
        # response = await websocket.recv()
        # print(f"Received: {response}")

asyncio.run(websocket_client())
