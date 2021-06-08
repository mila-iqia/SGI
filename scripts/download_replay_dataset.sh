#!/bin/bash
games='AirRaid Alien Amidar Assault Asterix Asteroids Atlantis BankHeist BattleZone BeamRider Berzerk Bowling Boxing Breakout Carnival Centipede ChopperCommand CrazyClimber DemonAttack DoubleDunk ElevatorAction Enduro FishingDerby Freeway Frostbite Gopher Gravitar Hero IceHockey Jamesbond JourneyEscape Kangaroo Krull KungFuMaster MontezumaRevenge MsPacman NameThisGame Phoenix Pitfall Pong Pooyan PrivateEye Qbert Riverraid RoadRunner Robotank Seaquest Skiing Solaris SpaceInvaders StarGunner Tennis TimePilot Tutankham UpNDown Venture VideoPinball WizardOfWor YarsRevenge Zaxxon'
ckpts='1 5 10 15 20 25 35 50'
runs='1 2 3 4 5'
files='action observation reward terminal'
export data_dir=$1

echo "Missing Files:"
for g in ${games[@]}; do
  for f in ${files[@]}; do
    for c in ${ckpts[@]}; do
      if [ ! -f "${data_dir}/${g}/${f}_${c}.gz" ]; then
        echo "${data_dir}/${g}/${f}_${c}.gz"
      fi;
    done;
    for r in ${runs[@]}; do
      if [ ! -f "${data_dir}/${g}/${f}_${r}01.gz" ]; then
        echo "${data_dir}/${g}/${f}_${r}01.gz"
      fi;
    done;
  done;
done;

# https://stackoverflow.com/a/226724
echo "Do you wish to download missing files?"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) break;;
        No ) exit;;
    esac
done

for g in ${games[@]}; do
  mkdir -p "${data_dir}/${g}"
  for f in ${files[@]}; do
    for c in ${ckpts[@]}; do
      if [ ! -f "${data_dir}/${g}/${f}_${c}.gz" ]; then
        gsutil cp "gs://atari-replay-datasets/dqn/${g}/1/replay_logs/\$store\$_${f}_ckpt.${c}.gz" "${data_dir}/${g}/${f}_${c}.gz"
      fi;
    done;
    for r in ${runs[@]}; do
      if [ ! -f "${data_dir}/${g}/${f}_${r}01.gz" ]; then
        gsutil cp "gs://atari-replay-datasets/dqn/${g}/${r}/replay_logs/\$store\$_${f}_ckpt.1.gz" "${data_dir}/${g}/${f}_${r}01.gz"
      fi;
    done;
  done;
done;
