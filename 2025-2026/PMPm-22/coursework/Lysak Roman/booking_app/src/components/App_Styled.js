import styled from "styled-components";
import symbolicsBackground from "../static/symbolics.png";

export const Wrapper = styled.div`
    display: flex;
    flex-direction: column;
    width: 100%;
    height: 100vh;
    overflow-y: hidden;
    position: relative;

    &::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image: url(${symbolicsBackground});
        background-size: contain;
        background-position: right;
        background-repeat: no-repeat;
        opacity: 0.3;
        pointer-events: none;
        z-index: 0;
    }
`

export const Header = styled.div`
    display: flex;
    flex-direction: row;
    padding: 40px 20px;
    justify-content: space-between;
    align-items: center;
`

export const UserInfoWrapper = styled.div`
    display: flex;
    flex-direction: column;
    align-items: end;
    gap: 5px;

    span {
        display: block;
        font-weight: bold;
    }
`

export const RightBlock = styled.div`
    display: flex;
    gap: 10px;
`

export const LeftBlock = styled.div`
    display: flex;
    flex-direction: column;
    gap: 5px;

    .img-wrapper {
        height: 60px;

        img {
            height: 100%;
        }
    }
`
