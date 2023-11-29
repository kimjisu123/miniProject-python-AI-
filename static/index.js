function camInit(stream) {
    var cameraView = document.getElementById("cameraview");
    cameraView.srcObject = stream;
    cameraView.play();
}

function camInitFailed(error) {
    console.log("get camera permission failed : ", error)
}

// Main init

function mainInit() {
    if(!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia )
    {
        alert("Media Device not supported")
        return;
    }
    navigator.mediaDevices.getUserMedia({video:true})
        .then(camInit)
        .catch(camInitFailed);
}

document.addEventListener('DOMContentLoaded', () => {

    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureBtn = document.getElementById('captureBtn');

    let stream;
    let photoData;

    // 미디어 스트림 가져오기
    navigator.mediaDevices.getUserMedia({ video: true })
        .then((mediaStream) => {
            video.srcObject = mediaStream;
            stream = mediaStream;
        })
        .catch((error) => {
            console.error('Error accessing webcam:', error);
        });

    // 사진 찍기 버튼 클릭 이벤트
    captureBtn.addEventListener('click', () => {
        // Canvas에 현재 비디오 프레임 그리기
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Canvas의 이미지 데이터를 Blob 형태로 변환
        canvas.toBlob((blob) => {
            // Blob을 FormData에 추가
            formData.append('photo', blob, 'photo.png');
            formData.append('additionalData', 'Additional Data');

            // 서버로 이미지 데이터 및 추가 데이터 전송
            sendFormDataToServer(formData);
        }, 'image/png');
    });

    // 서버로 이미지 데이터 및 추가 데이터 전송
    function sendFormDataToServer(formData) {
        // Fetch API를 사용하여 FormData 전송
        fetch('/upload', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => console.log(data))
        .catch(error => console.error('Error:', error));

        alert(result.message);
    }
    window.addEventListener('beforeunload', () => {
        if (stream) {
            const tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
        }
    });
});

