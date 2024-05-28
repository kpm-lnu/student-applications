render_ecs: __build_export_ecs __build_render __turbulence __render

profile_ecs: __build_profile_ecs __turbulence __profile

render_oop: __build_export_oop __build_render __turbulence __render

profile_oop: __build_profile_oop __turbulence __profile

test_ecs:
	@python3 ./test/main.py --model=ecs

test_oop:
	@python3 ./test/main.py --model=oop

__init:
	@make __build_init
	@mkdir -p ./buff
	@./bin/init

__build_init:
	@go build -o ./bin/init ./init

__build_export_ecs:
	@go build -o ./bin/turbulence -tags BUILD_ECS,BUILD_EXPORT ./main/turbulence

__build_profile_ecs:
	@go build -o ./bin/turbulence -tags BUILD_ECS,BUILD_PROFILE ./main/turbulence

__build_export_oop:
	@go build -o ./bin/turbulence -tags BUILD_OOP,BUILD_EXPORT ./main/turbulence

__build_profile_oop:
	@go build -o ./bin/turbulence -tags BUILD_OOP,BUILD_PROFILE ./main/turbulence

__build_render:
	@go build -o ./bin/graphics ./main/render

__turbulence:
	@./bin/turbulence

__render:
	@./bin/graphics

__profile:
	@go tool pprof -png ./bin/turbulence ./buff/cpu-profile > ./buff/cpu-profile.png
	@go tool pprof -png ./bin/turbulence ./buff/memory-profile > ./buff/memory-profile.png
