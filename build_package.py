#!/usr/bin/env python3
"""
S3 Uploader Package 빌드 및 배포 스크립트
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, check=True, capture_output=False):
    """명령어 실행"""
    print(f"실행 중: {command}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=check, 
            capture_output=capture_output,
            text=True
        )
        if capture_output:
            return result.stdout.strip()
        return result
    except subprocess.CalledProcessError as e:
        print(f"명령어 실행 실패: {e}")
        if capture_output and e.stdout:
            print(f"출력: {e.stdout}")
        if capture_output and e.stderr:
            print(f"오류: {e.stderr}")
        sys.exit(1)

def clean_build():
    """빌드 디렉토리 정리"""
    print("빌드 디렉토리 정리 중...")
    dirs_to_clean = ['build', 'dist', '*.egg-info']
    
    for pattern in dirs_to_clean:
        for path in Path('.').glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"삭제됨: {path}")
            elif path.is_file():
                path.unlink()
                print(f"삭제됨: {path}")

def check_dependencies():
    """의존성 확인"""
    print("의존성 확인 중...")
    required_packages = [
        'setuptools',
        'wheel',
        'build',
        'twine'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"설치 필요한 패키지: {missing_packages}")
        install_cmd = f"pip install {' '.join(missing_packages)}"
        run_command(install_cmd)
    else:
        print("모든 의존성이 설치되어 있습니다.")

def build_package():
    """패키지 빌드"""
    print("패키지 빌드 중...")
    
    # 소스 배포판 빌드
    run_command("python -m build --sdist")
    
    # 휠 배포판 빌드
    run_command("python -m build --wheel")
    
    print("빌드 완료!")

def check_package():
    """패키지 검사"""
    print("패키지 검사 중...")
    
    # dist 디렉토리 확인
    dist_dir = Path('dist')
    if not dist_dir.exists():
        print("오류: dist 디렉토리가 없습니다.")
        return False
    
    files = list(dist_dir.glob('*'))
    if not files:
        print("오류: dist 디렉토리에 파일이 없습니다.")
        return False
    
    print(f"빌드된 파일들:")
    for file in files:
        print(f"  - {file.name}")
    
    return True

def test_install():
    """테스트 설치"""
    print("테스트 설치 중...")
    
    # 임시 디렉토리에서 설치 테스트
    test_dir = Path('test_install')
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    test_dir.mkdir()
    
    try:
        # 가상환경 생성
        run_command(f"python -m venv {test_dir}/venv")
        
        # 가상환경 활성화 및 설치
        if os.name == 'nt':  # Windows
            activate_cmd = f"{test_dir}/venv/Scripts/activate"
        else:  # Unix/Linux/macOS
            activate_cmd = f"source {test_dir}/venv/bin/activate"
        
        # 최신 빌드 파일 찾기
        dist_files = list(Path('dist').glob('*.whl'))
        if not dist_files:
            print("오류: 휠 파일을 찾을 수 없습니다.")
            return False
        
        latest_wheel = max(dist_files, key=lambda x: x.stat().st_mtime)
        
        # 설치 테스트
        install_cmd = f"{activate_cmd} && pip install {latest_wheel}"
        run_command(install_cmd)
        
        # 설치 확인
        check_cmd = f"{activate_cmd} && python -c 'import s3_uploader; print(s3_uploader.__version__)'"
        version = run_command(check_cmd, capture_output=True)
        print(f"설치된 버전: {version}")
        
        print("테스트 설치 성공!")
        return True
        
    except Exception as e:
        print(f"테스트 설치 실패: {e}")
        return False
    finally:
        # 정리
        if test_dir.exists():
            shutil.rmtree(test_dir)

def upload_to_testpypi():
    """TestPyPI에 업로드"""
    print("TestPyPI에 업로드 중...")
    
    # 업로드 전 검사
    run_command("twine check dist/*")
    
    # TestPyPI 업로드
    run_command("twine upload --repository testpypi dist/*")
    
    print("TestPyPI 업로드 완료!")

def upload_to_pypi():
    """PyPI에 업로드"""
    print("PyPI에 업로드 중...")
    
    # 업로드 전 검사
    run_command("twine check dist/*")
    
    # PyPI 업로드
    run_command("twine upload dist/*")
    
    print("PyPI 업로드 완료!")

def main():
    """메인 함수"""
    print("=== S3 Uploader Package 빌드 및 배포 ===")
    
    # 현재 디렉토리 확인
    if not Path('setup.py').exists():
        print("오류: setup.py 파일이 없습니다. 올바른 디렉토리에서 실행하세요.")
        sys.exit(1)
    
    # 명령행 인자 처리
    if len(sys.argv) < 2:
        print("사용법:")
        print("  python build_package.py clean     # 빌드 디렉토리 정리")
        print("  python build_package.py build     # 패키지 빌드")
        print("  python build_package.py test      # 테스트 설치")
        print("  python build_package.py testpypi  # TestPyPI 업로드")
        print("  python build_package.py pypi      # PyPI 업로드")
        print("  python build_package.py all       # 전체 과정")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "clean":
        clean_build()
    elif command == "build":
        check_dependencies()
        clean_build()
        build_package()
        check_package()
    elif command == "test":
        if not check_package():
            print("먼저 패키지를 빌드하세요: python build_package.py build")
            sys.exit(1)
        test_install()
    elif command == "testpypi":
        if not check_package():
            print("먼저 패키지를 빌드하세요: python build_package.py build")
            sys.exit(1)
        upload_to_testpypi()
    elif command == "pypi":
        if not check_package():
            print("먼저 패키지를 빌드하세요: python build_package.py build")
            sys.exit(1)
        upload_to_pypi()
    elif command == "all":
        check_dependencies()
        clean_build()
        build_package()
        if check_package() and test_install():
            print("\n전체 과정 완료!")
            print("PyPI에 업로드하려면: python build_package.py pypi")
    else:
        print(f"알 수 없는 명령어: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main() 