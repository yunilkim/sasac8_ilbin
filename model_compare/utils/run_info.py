"""
실행 직전에 어떤 설정으로 도는지 화면에 출력

폴더명 확인까지 받을지는 config/run_config.py 의 CONFIRM_RUN 으로 정함
    CONFIRM_RUN = False : 설정만 보여주고 바로 진행 (기본, 백그라운드 실행에 안전)
    CONFIRM_RUN = True  : 만들어질 폴더명을 입력란에 띄우고 Enter 또는 수정 입력
"""

# 설정 dict 출력
def print_settings(title, settings):
    print()
    print(f'===== {title} =====')

    for key, value in settings.items():
        print(f'  {key:<16} : {value}')


# CONFIRM_RUN = True 일때 실행 모델 프로젝트 저장명 입력
def ask_run_name(run_name):
    try:
        # 기본값을 괄호 안에 보여준다. Enter 만 치면 기본값 사용.
        typed = input(f'  만들어질 폴더명 [{run_name}] : ').strip()
    except (EOFError, OSError):
        print('  (입력을 받을 수 없어 기본 폴더명으로 진행합니다)')
        return run_name

    return typed if typed else run_name

# CONFIRM_RUN = True 일때 최종 확인
def ask_yes(question='  이대로 진행할까요? (y/n) : '):
    """진행 여부를 묻는다. 입력을 받을 수 없으면 진행으로 본다."""
    try:
        answer = input(question).strip().lower()
    except (EOFError, OSError):
        return True

    # 그냥 Enter 를 쳐도 진행으로 본다
    return answer in ('', 'y', 'yes')

# 확인 실행 
def confirm_run(title, settings, run_name=None, confirm=False):
    """
    title    : 화면에 표시할 제목
    settings : 보여줄 설정 딕셔너리
    run_name : 만들어질 폴더명 (없으면 폴더명 입력은 건너뛴다)
    confirm  : True 면 입력을 받는다

    반환값 : (진행할지 여부, 최종 폴더명)
    """
    print_settings(title, settings)

    if not confirm:
        return True, run_name

    if run_name is not None:
        run_name = ask_run_name(run_name)

    return ask_yes(), run_name
